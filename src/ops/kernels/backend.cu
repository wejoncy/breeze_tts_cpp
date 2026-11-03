#include "ops/backend.h"
#include "ops/cuda/blas_check.h"
#include "ops/cuda/lt_linear.h"
#include "ops/kernels/common/cuda_math.cuh"
#include "ops/kernels/common/chip_execution.h"
#include "ops/attention.h"
#include "ops/kernels/attention/cache_address.cuh"
#include "ops/kernels/attention/qkv_writeback.cuh"
#include "ops/sequence.h"
#include "ops/kernels/sequence/norm_projection.h"
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <initializer_list>

namespace tts {
__device__ inline void dependency_wait() {
#if __CUDA_ARCH__ >= 900
    cudaGridDependencySynchronize();
#endif
}
__device__ inline void dependency_release() {
#if __CUDA_ARCH__ >= 900
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}
static cudaLaunchConfig_t pdl_launch(dim3 grid, int threads, cudaStream_t stream,
                                     cudaLaunchAttribute &attr) {
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = grid;
    cfg.blockDim = dim3(threads);
    cfg.stream = stream;
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    return cfg;
}
static bool compiled_pdl_support();
static std::array<int, 4> gemv_capacities(int multiprocessors);

__device__ float block_sum(float x) {
    __shared__ float s[32];
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    x = warp_sum(x);
    if (!lane)
        s[warp] = x;
    __syncthreads();
    x = threadIdx.x < blockDim.x / 32 ? s[lane] : 0;
    if (!warp)
        x = warp_sum(x);
    return x;
}
Backend::Backend(int device, Quantization quantization, size_t memory_limit_bytes)
    : memory(device, memory_limit_bytes), device(device), stream(memory.stream),
      blas_workspace_(8 << 20, &memory) {
    try {
        CUDA(cudaSetDevice(device));
        cudaDeviceProp p{};
        CUDA(cudaGetDeviceProperties(&p, device));
        shared_memory_per_block_ = p.sharedMemPerBlock;
        multiprocessors_ = p.multiProcessorCount;
        normalized_linear_capacity_ = ops::normalized_linear_capacity(
            multiprocessors_, fp32_latency_warps(p.major, p.minor), normalized_qkv_capacity_);
        if (p.major < 8)
            throw std::runtime_error("BF16 engine requires SM80+");
        BLAS(cublasCreate(&blas));
        BLAS(cublasSetStream(blas, stream));
        BLAS(cublasSetWorkspace(blas, blas_workspace_.data(), blas_workspace_.size()));
        BLAS(cublasSetMathMode(blas, CUBLAS_TF32_TENSOR_OP_MATH));
        BLAS(cublasLtCreate(&lt_));
        lt_linear_ = std::make_unique<LtLinear>(lt_, blas);
        initialize_flash_attention();
        for (auto type : {Type::BF16, Type::F16})
            for (int dim : {32, 64, 96, 128}) {
                int resident = flash_decode_resident_blocks(dim, type);
                int64_t slots = int64_t(multiprocessors_) * resident;
                if (slots > INT_MAX) throw std::overflow_error("decode resident CTA capacity");
                flash_decode_slots_[(type == Type::F16 ? 4 : 0) + dim / 32 - 1] = int(slots);
            }
        gemv_slots_ = gemv_capacities(multiprocessors_);
        qkv_gemv_capacity_ = std::min(gemv_slots_[1], gemv_slots_[2]);
        if (quantization != Quantization::None)
            quant_ = std::make_unique<QuantBackend>(memory, lt_, blas_workspace_,
                                                   quantized_type(quantization));
        pdl_ = p.major >= 9 && compiled_pdl_support();
    } catch (...) {
        quant_.reset();
        lt_linear_.reset();
        if (lt_)
            cublasLtDestroy(lt_);
        if (blas)
            cublasDestroy(blas);
        throw;
    }
}
Backend::~Backend() {
    if (stream)
        cudaStreamSynchronize(stream);
    quant_.reset();
    lt_linear_.reset();
    if (lt_)
        cublasLtDestroy(lt_);
    if (blas)
        cublasDestroy(blas);
}

// Four independent rows per CTA; contiguous vector loads coalesce weight reads.
template <class T, int WARPS = 4, bool PDL = false>
__global__ void gemv_kernel(const T *__restrict__ x, const T *__restrict__ w, T *__restrict__ y,
                            int n, int k, int batch, int xstride, int ystride,
                            QkvProjectionWriteback writeback) {
    int row = blockIdx.x * WARPS + threadIdx.x / 32, b = blockIdx.y, lane = threadIdx.x % 32;
    // Independent weight prefetch may overlap the producer's reduction/epilogue.
    float prefetched[32];
    if constexpr (PDL) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                int c = lane * 4 + j * 128 + v;
                prefetched[j * 4 + v] = row < n && c < k ? f(w[size_t(row) * k + c]) : 0.f;
            }
        }
        dependency_wait();
        dependency_release();
    }
    float sum = 0;
    if (row < n) {
        if constexpr (PDL) {
            for (int c = lane * 4; c < k; c += 128) {
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    if (c + j < k)
                        sum = fmaf(prefetched[(c - lane * 4) / 128 * 4 + j],
                                   f(x[size_t(b) * xstride + c + j]), sum);
            }
        } else {
            sum = warp_dot_four(x + size_t(b) * xstride, w + size_t(row) * k, k, lane);
        }
    }
    sum = warp_sum(sum);
    if (row < n && !lane) {
        T projected = r<T>(sum);
        if (writeback.query)
            writeback.store(projected, b, row);
        else
            y[size_t(b) * ystride + row] = projected;
    }
}
static std::array<int, 4> gemv_capacities(int multiprocessors) {
    std::array<int, 4> blocks{};
    CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks[0], gemv_kernel<float, 4, false>, 128, 0));
    CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks[1], gemv_kernel<__half, 4, false>, 128, 0));
    CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks[2], gemv_kernel<bf16, 4, false>, 128, 0));
    CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks[3], gemv_kernel<bf16, 4, true>, 128, 0));
    for (auto &count : blocks) count *= multiprocessors;
    return blocks;
}

static bool compiled_pdl_support() {
    cudaFuncAttributes compiled{};
    CUDA(cudaFuncGetAttributes(&compiled, gemv_kernel<__nv_bfloat16, 4, true>));
    return compiled.ptxVersion >= 90;
}
void Backend::linear(TensorView x, TensorView w, TensorView y, bool full_precision, float accumulation,
                       float product_scale) {
    if (x.cols != w.cols || y.rows != x.rows || y.cols != w.rows || x.type != w.type ||
        (x.type != y.type && !((x.type == Type::BF16 || x.type == Type::F16) && y.type == Type::F32)) ||
        !w.dense() || (x.type != Type::BF16 && x.type != Type::F16 && x.type != Type::F32))
        throw std::runtime_error("linear dimensions/dtype mismatch");
    bool dependent = pdl_ && x.type == Type::BF16 && w.cols <= 1024;
    int capacity = gemv_slots_[x.type == Type::F32 ? 0 : x.type == Type::F16 ? 1 : dependent ? 3 : 2];
    int64_t blocks = ((int64_t(w.rows) + 3) / 4) * x.rows;
    // Independent row reductions use GEMV while the complete grid fits one
    // residency wave. Larger grids use the prepared matrix-multiply plans.
    if (!accumulation && product_scale == 1 && x.type == y.type && blocks <= capacity) {
        dim3 grid((w.rows + 3) / 4, x.rows);
        if (dependent) {
            cudaLaunchAttribute attr{};
            auto cfg = pdl_launch(grid, 128, stream, attr);
            CUDA(cudaLaunchKernelEx(&cfg, gemv_kernel<bf16, 4, true>, x.ptr<bf16>(), w.ptr<bf16>(),
                                    y.ptr<bf16>(), w.rows, w.cols, x.rows, x.ld(), y.ld(), QkvProjectionWriteback{}));
        } else if (x.type == Type::BF16)
            gemv_kernel<<<grid, 128, 0, stream>>>(x.ptr<bf16>(), w.ptr<bf16>(), y.ptr<bf16>(),
                                                  w.rows, w.cols, x.rows, x.ld(), y.ld(), QkvProjectionWriteback{});
        else if (x.type == Type::F16)
            gemv_kernel<<<grid, 128, 0, stream>>>(x.ptr<__half>(), w.ptr<__half>(), y.ptr<__half>(),
                                                w.rows, w.cols, x.rows, x.ld(), y.ld(), QkvProjectionWriteback{});
        else
            gemv_kernel<<<grid, 128, 0, stream>>>(x.ptr<float>(), w.ptr<float>(), y.ptr<float>(),
                                                  w.rows, w.cols, x.rows, x.ld(), y.ld(), QkvProjectionWriteback{});
        return;
    }
    if (!lt_linear_->linear(x, w, y, full_precision, accumulation, product_scale,
                             blas_workspace_.data(), blas_workspace_.size(), stream))
        throw std::invalid_argument("unsupported prepared Lt GEMM views");
}
bool Backend::linear_qkv(TensorView x, TensorView w, const QkvProjectionOutput &o) {
    o.validate(x, w);
    int64_t blocks = ((int64_t(w.rows) + 3) / 4) * x.rows;
    if (blocks > qkv_gemv_capacity_)
        return false;
    dim3 grid((w.rows + 3) / 4, x.rows);
    QkvProjectionWriteback output(o);
    if (x.type == Type::F16)
        gemv_kernel<__half, 4, false><<<grid, 128, 0, stream>>>(
            x.ptr<__half>(), w.ptr<__half>(), nullptr, w.rows, w.cols, x.rows, x.ld(), 0, output);
    else
        gemv_kernel<bf16, 4, false><<<grid, 128, 0, stream>>>(
            x.ptr<bf16>(), w.ptr<bf16>(), nullptr, w.rows, w.cols, x.rows, x.ld(), 0, output);
    CUDA(cudaGetLastError());
    return true;
}
template <class T>
__global__ void norm_kernel(const T *x, const T *w, const T *bias, T *y, int cols, float eps,
                            bool gemma) {
    int row = blockIdx.x;
    float ss = 0, mean = 0;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = f(x[size_t(row) * cols + c]);
        ss += v * v;
        mean += v;
    }
    __shared__ float inv, mu;
    float sum = block_sum(mean);
    if (!threadIdx.x)
        mu = bias ? sum / cols : 0;
    __syncthreads();
    sum = block_sum(ss);
    if (!threadIdx.x)
        inv = rsqrtf(fmaxf(sum / cols - mu * mu, 0.f) + eps);
    __syncthreads();
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float z = (f(x[size_t(row) * cols + c]) - mu) * inv;
        if (!gemma && !bias)
            z = rounded<T>(z);
        y[size_t(row) * cols + c] =
            r<T>(z * (f(w[c]) + (gemma ? 1.f : 0.f)) + (bias ? f(bias[c]) : 0.f));
    }
}
template <class T, bool ADD>
__global__ void rms_kernel(T *x, const T *update, const T *w, T *out, const T *scale, int cols,
                           float eps, bool gemma) {
    __shared__ float sums[8];
    int row = blockIdx.x;
    float ss = 0;
    // All supported RMS widths <= 2048; retain values across the reduction.
    float values[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
        int c = threadIdx.x + j * 256;
        float z = 0;
        if (c < cols) {
            z = f(x[size_t(row) * cols + c]);
            if (ADD) {
                z = rounded<T>(z + rounded<T>(f(update[size_t(row) * cols + c]) *
                                              (scale ? f(scale[c]) : 1.f)));
                x[size_t(row) * cols + c] = r<T>(z);
            }
        }
        values[j] = z;
        ss += z * z;
    }
    ss = warp_sum(ss);
    if (threadIdx.x % 32 == 0)
        sums[threadIdx.x / 32] = ss;
    __syncthreads();
    float total = 0;
#pragma unroll
    for (int j = 0; j < 8; j++)
        total += sums[j];
    float inv = rsqrtf(total / cols + eps);
#pragma unroll
    for (int j = 0; j < 8; j++) {
        int c = threadIdx.x + j * 256;
        if (c < cols) {
            float z = values[j] * inv;
            if (!gemma)
                z = rounded<T>(z);
            out[size_t(row) * cols + c] = r<T>(z * (f(w[c]) + (gemma ? 1.f : 0.f)));
        }
    }
}
template <int WIDTH, bool ADD>
__global__ void rms_bf16_kernel(bf16 *x, const bf16 *update, const bf16 *w, bf16 *out,
                                const bf16 *scale, float eps, bool gemma, const int *ends) {
    constexpr int THREADS = WIDTH == 2048 ? 256 : 128, ITEMS = (WIDTH / 2 + THREADS - 1) / THREADS;
    __shared__ float sums[THREADS / 32];
    int row = ends ? ends[blockIdx.x] - 1 : blockIdx.x;
    float ss = 0;
    float2 values[ITEMS];
    const auto *xp = reinterpret_cast<const __nv_bfloat162 *>(x + size_t(row) * WIDTH);
    const auto *up =
        reinterpret_cast<const __nv_bfloat162 *>(update ? update + size_t(row) * WIDTH : nullptr);
    const auto *wp = reinterpret_cast<const __nv_bfloat162 *>(w);
#pragma unroll
    for (int j = 0; j < ITEMS; j++) {
        int c = threadIdx.x + j * THREADS;
        float2 z = make_float2(0, 0);
        if (c < WIDTH / 2) {
            z = __bfloat1622float2(xp[c]);
            if (ADD) {
                float2 d = __bfloat1622float2(up[c]);
                if (scale) {
                    float2 sc =
                        __bfloat1622float2(reinterpret_cast<const __nv_bfloat162 *>(scale)[c]);
                    d.x = rounded<bf16>(d.x * sc.x);
                    d.y = rounded<bf16>(d.y * sc.y);
                }
                auto sum = __floats2bfloat162_rn(z.x + d.x, z.y + d.y);
                reinterpret_cast<__nv_bfloat162 *>(x + size_t(row) * WIDTH)[c] = sum;
                z = __bfloat1622float2(sum);
            }
        }
        values[j] = z;
        ss += z.x * z.x + z.y * z.y;
    }
    dependency_release();
    ss = warp_sum(ss);
    if (threadIdx.x % 32 == 0)
        sums[threadIdx.x / 32] = ss;
    __syncthreads();
    float total = 0;
#pragma unroll
    for (int j = 0; j < THREADS / 32; j++)
        total += sums[j];
    float inv = rsqrtf(total / WIDTH + eps);
#pragma unroll
    for (int j = 0; j < ITEMS; j++) {
        int c = threadIdx.x + j * THREADS;
        if (c < WIDTH / 2) {
            float2 z = values[j], weight = __bfloat1622float2(wp[c]);
            z.x *= inv;
            z.y *= inv;
            if (!gemma) {
                z = __bfloat1622float2(__floats2bfloat162_rn(z.x, z.y));
            }
            reinterpret_cast<__nv_bfloat162 *>(out + size_t(blockIdx.x) * WIDTH)[c] =
                __floats2bfloat162_rn(z.x * (weight.x + (gemma ? 1.f : 0.f)),
                                      z.y * (weight.y + (gemma ? 1.f : 0.f)));
        }
    }
}
template <bool ADD>
static bool fast_rms(cudaStream_t stream, TensorView x, TensorView u, TensorView w,
                     TensorView out, TensorView scale, float eps, bool gemma,
                     const int *ends = nullptr) {
    if (x.type != Type::BF16)
        return false;
#define RMS_WIDTH(N, T)                                                                            \
    case N: {                                                                                      \
        rms_bf16_kernel<N, ADD><<<(ends ? out.rows : x.rows), T, 0, stream>>>(                        \
            x.ptr<bf16>(), u.ptr<bf16>(), w.ptr<bf16>(), out.ptr<bf16>(), scale.ptr<bf16>(),           \
            eps, gemma, ends);                                                                     \
        return true;                                                                               \
    }
    switch (x.cols) {
        RMS_WIDTH(1024, 128) RMS_WIDTH(2048, 256) RMS_WIDTH(1152, 128) default : return false;
    }
#undef RMS_WIDTH
}
static void check_float_storage(TensorView source, std::initializer_list<TensorView> operands) {
    if (source.type != Type::BF16 && source.type != Type::F16 && source.type != Type::F32)
        throw std::invalid_argument("operator requires F16, BF16 or F32 storage");
    for (auto operand : operands)
        if (operand.data && operand.type != source.type)
            throw std::invalid_argument("operator storage types do not match");
}
template <class Launch> static void dispatch_float_storage(Type type, Launch &&launch) {
    switch (type) {
    case Type::BF16: launch(bf16{}); break;
    case Type::F16: launch(__half{}); break;
    case Type::F32: launch(float{}); break;
    default: throw std::invalid_argument("operator requires F16, BF16 or F32 storage");
    }
}
void Backend::norm(TensorView x, TensorView w, TensorView y, float eps, bool gemma,
                   TensorView bias) {
    check_float_storage(x, {w, y, bias});
    if (!x.dense() || !y.dense() || !w.dense() || x.rows != y.rows || x.cols != y.cols ||
        x.type != y.type || w.numel() != size_t(x.cols) || w.type != x.type ||
        (bias.data && (!bias.dense() || bias.numel() != size_t(x.cols))))
        throw std::invalid_argument("norm shape/dtype/layout mismatch");
    if (!bias.data && fast_rms<false>(stream, x, {}, w, y, {}, eps, gemma))
        return;
    dispatch_float_storage(x.type, [&](auto storage) {
        using T = decltype(storage);
        if (!bias.data && x.cols <= 2048)
            rms_kernel<T, false><<<x.rows, 256, 0, stream>>>(
                x.ptr<T>(), nullptr, w.ptr<T>(), y.ptr<T>(), nullptr, x.cols, eps, gemma);
        else
            norm_kernel<<<x.rows, 256, 0, stream>>>(x.ptr<T>(), w.ptr<T>(), bias.ptr<T>(),
                                                   y.ptr<T>(), x.cols, eps, gemma);
    });
}
void Backend::residual_norm(TensorView x, TensorView update, TensorView w, TensorView out,
                            float eps, TensorView scale, bool gemma) {
    check_float_storage(x, {update, w, out, scale});
    if (fast_rms<true>(stream, x, update, w, out, scale, eps, gemma))
        return;
    if (x.cols > 2048)
        throw std::runtime_error("fused RMS width exceeds 2048");
    dispatch_float_storage(x.type, [&](auto storage) {
        using T = decltype(storage);
        rms_kernel<T, true><<<x.rows, 256, 0, stream>>>(x.ptr<T>(), update.ptr<T>(),
            w.ptr<T>(), out.ptr<T>(), scale.ptr<T>(), x.cols, eps, gemma);
    });
}
void Backend::residual_norm_last(TensorView x, TensorView u, TensorView w, TensorView out,
                                 const int *ends, float eps) {
    check_float_storage(x, {u, w, out});
    if (x.type != Type::BF16)
        throw std::invalid_argument("packed final norm requires BF16 storage");
    if (!fast_rms<true>(stream, x, u, w, out, {}, eps, false, ends))
        throw std::runtime_error("unsupported packed final norm");
}
template <class T>
__global__ void add_kernel(const T *a, const T *b, T *y, const T *scale, size_t n, int cols) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = r<T>(f(a[i]) + rounded<T>(f(b[i]) * (scale ? f(scale[i % cols]) : 1.f)));
}
void Backend::add(TensorView a, TensorView b, TensorView out, TensorView scale) {
    check_float_storage(a, {b, out, scale});
    size_t n = size_t(a.rows) * a.cols;
    dispatch_float_storage(a.type, [&](auto storage) {
        using T = decltype(storage);
        add_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
            a.ptr<T>(), b.ptr<T>(), out.ptr<T>(), scale.ptr<T>(), n, a.cols);
    });
}
template <class T> __global__ void bias_kernel(T *x, const T *b, size_t n, int cols) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] = r<T>(f(x[i]) + f(b[i % cols]));
}
void Backend::bias(TensorView x, TensorView b) {
    check_float_storage(x, {b});
    if (!b.data)
        return;
    size_t n = size_t(x.rows) * x.cols;
    dispatch_float_storage(x.type, [&](auto storage) {
        using T = decltype(storage);
        bias_kernel<<<(n + 255) / 256, 256, 0, stream>>>(x.ptr<T>(), b.ptr<T>(), n, x.cols);
    });
}
__device__ float act(float x, int kind) {
    if (kind == 3)
        return x >= 0 ? x : expm1f(x);
    if (kind == 0)
        return x / (1.f + expf(-x));
    if (kind == 1)
        return .5f * x * (1.f + tanhf(.7978845608028654f * (x + .044715f * x * x * x)));
    return .5f * x * (1.f + erff(x * .7071067811865475f));
}
template <class T> __global__ void activation_kernel(const T *x, T *y, size_t n, int kind) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = r<T>(act(f(x[i]), kind));
}
void Backend::activation(TensorView x, TensorView y, int kind) {
    check_float_storage(x, {y});
    size_t n = size_t(x.rows) * x.cols;
    dispatch_float_storage(x.type, [&](auto storage) {
        using T = decltype(storage);
        activation_kernel<<<(n + 255) / 256, 256, 0, stream>>>(x.ptr<T>(), y.ptr<T>(), n, kind);
    });
}
template <class T> __global__ void gated_kernel(const T *gu, T *y, size_t n, int cols, int kind) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        size_t p = (i / cols) * cols * 2 + i % cols;
        y[i] = r<T>(rounded<T>(act(f(gu[p]), kind)) * f(gu[p + cols]));
    }
}
void Backend::gated(TensorView gu, TensorView y, int kind) {
    check_float_storage(y, {gu});
    size_t n = size_t(y.rows) * y.cols;
    dispatch_float_storage(y.type, [&](auto storage) {
        using T = decltype(storage);
        gated_kernel<<<(n + 255) / 256, 256, 0, stream>>>(gu.ptr<T>(), y.ptr<T>(), n, y.cols, kind);
    });
}
template <class T>
__global__ void embedding_kernel(const T *w, const int *ids, T *out, int rows, int cols,
                                 float scale, const T *replacement, int replacement_id) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size_t(rows) * cols) {
        int id = ids[i / cols];
        out[i] = (id == replacement_id && replacement)
                     ? replacement[i % cols]
                     : r<T>(f(w[size_t(id) * cols + i % cols]) * rounded<T>(scale));
    }
}
void Backend::embedding(TensorView table, const int *ids, TensorView out, float scale,
                        TensorView replacement, int replacement_id) {
    check_float_storage(out, {table, replacement});
    size_t n = size_t(out.rows) * out.cols;
    dispatch_float_storage(out.type, [&](auto storage) {
        using T = decltype(storage);
        embedding_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
            table.ptr<T>(), ids, out.ptr<T>(), out.rows, out.cols, scale,
            replacement.ptr<T>(), replacement_id);
    });
}
template <class T>
__global__ void rope_table_kernel(T *table, int rows, int dim, float theta, float factor,
                                  int rope_type, float low_wave, float high_wave) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * dim / 2)
        return;
    int p = i / (dim / 2), d = i % (dim / 2);
    float freq = powf(theta, -2.f * d / dim);
    if (rope_type == 1)
        freq /= factor;
    if (rope_type == 2) {
        float wave = 6.28318530718f / freq;
        if (wave > low_wave)
            freq /= factor;
        else if (wave >= high_wave) {
            float smooth = (low_wave / wave - 1) / (low_wave / high_wave - 1);
            freq = (1 - smooth) * freq / factor + smooth * freq;
        }
    }
    table[p * dim + d] = r<T>(cosf(p * freq));
    table[p * dim + d + dim / 2] = r<T>(sinf(p * freq));
}
void Backend::make_rope(TensorView table, int dim, float theta, float factor, int rope_type,
                        float low_wave, float high_wave) {
    check_float_storage(table, {});
    if (rope_type == 2 && !(low_wave > high_wave && high_wave > 0))
        throw std::invalid_argument("invalid Llama3 RoPE wavelength bounds");
    int n = table.rows * dim / 2;
    dispatch_float_storage(table.type, [&](auto storage) {
        using T = decltype(storage);
        rope_table_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
            table.ptr<T>(), table.rows, dim, theta, factor, rope_type, low_wave, high_wave);
    });
}
template <class T, int Mode>
__global__ void qkv_kernel(const T *in, T *q, T *ck, T *cv, const int *positions, int time,
                           int slot, int cap, int nh, int nk, int dim, const T *qn, const T *kn,
                           float eps, bool gemma, const T *rope, int fixed_position,
                           const int *row_batch, const int *row_position, CacheLayout cache,
                           const T *bias, int input_stride, int query_stride) {
    int row = blockIdx.x, h = blockIdx.y, t = row_position ? row_position[row] : row % time,
        b = row_batch ? row_batch[row] : row / time,
        p = row_position ? t : 0;
    if (b < 0 || t < 0)
        return;
    p = row_position ? t : (positions ? positions[b] : fixed_position) + t;
    if (cache.sequences) {
        const auto &sequence = cache.sequences[b];
        if (!sequence.position)
            return;
        if (!cache.key_pages) {
            ck = reinterpret_cast<T *>(static_cast<char *>(sequence.keys) +
                                       cache.layer * sequence.layer_bytes);
            cv = reinterpret_cast<T *>(static_cast<char *>(sequence.values) +
                                       cache.layer * sequence.layer_bytes);
        }
        p = *sequence.position + t;
        cache.block_table = sequence.blocks;
        cache.block_stride = 0;
        b = slot = 0;
    }
    bool is_q = h < nh;
    int local = is_q ? h : h - nh;
    const T *norm = is_q ? qn : kn;
    const T *src = in + size_t(row) * input_stride + h * dim;
    const T *head_bias = bias ? bias + h * dim : nullptr;
    T *dst =
        is_q ? q + size_t(row) * query_stride + local * dim
             : cache_pointer<T, Mode>(ck, b + slot, p, cap, nk * dim, cache, false) + local * dim;
    // Affine-only projections need neither a normalization reduction nor rotary
    // loads. Bias and the persistent KV write share this projection epilogue.
    if (!norm && !rope) {
        for (int d = threadIdx.x; d < dim; d += blockDim.x)
            dst[d] = head_bias ? r<T>(f(src[d]) + f(head_bias[d])) : src[d];
    } else {
    __shared__ float inv;
    float ss = 0;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float v = head_bias ? rounded<T>(f(src[d]) + f(head_bias[d])) : f(src[d]);
        ss += v * v;
    }
    float sum = block_sum(ss);
    if (!threadIdx.x)
        inv = norm ? rsqrtf(sum / dim + eps) : 1;
    __syncthreads();
    for (int d = threadIdx.x; d < dim / 2; d += blockDim.x) {
        float a = f(src[d]), z = f(src[d + dim / 2]);
        if (head_bias) {
            a = rounded<T>(a + f(head_bias[d]));
            z = rounded<T>(z + f(head_bias[d + dim / 2]));
        }
        if (norm) {
            a = a * inv;
            z = z * inv;
            if (!gemma) {
                a = rounded<T>(a);
                z = rounded<T>(z);
            }
            a = rounded<T>(a * (f(norm[d]) + (gemma ? 1 : 0)));
            z = rounded<T>(z * (f(norm[d + dim / 2]) + (gemma ? 1 : 0)));
        }
        if (rope) {
            float co = f(rope[p * dim + d]), si = f(rope[p * dim + d + dim / 2]);
            dst[d] = r<T>(rounded<T>(a * co) - rounded<T>(z * si));
            dst[d + dim / 2] = r<T>(rounded<T>(z * co) + rounded<T>(a * si));
        } else {
            dst[d] = r<T>(a);
            dst[d + dim / 2] = r<T>(z);
        }
    }
    }
    if (!is_q) {
        const T *v = in + size_t(row) * input_stride + (nh + nk + local) * dim;
        const T *vb = bias ? bias + (nh + nk + local) * dim : nullptr;
        T *vd = cache_pointer<T, Mode>(cv, b + slot, p, cap, nk * dim, cache, true) + local * dim;
        for (int d = threadIdx.x; d < dim; d += blockDim.x)
            vd[d] = vb ? r<T>(f(v[d]) + f(vb[d])) : v[d];
    }
}
template <class T, int Mode>
static void launch_qkv(cudaStream_t stream, TensorView in, TensorView q, TensorView k, TensorView v,
                       const int *pos, int batch, int time, int slot, int cap, int nh, int nk,
                       int dim, TensorView qn, TensorView kn, float eps, bool gemma,
                       TensorView rope, int fixed_position, const SequenceLayout *packed,
                       CacheLayout cache, TensorView bias) {
    dim3 grid(packed ? packed->rows : batch * time, nh + nk);
    qkv_kernel<T, Mode><<<grid, 128, 0, stream>>>(
        in.ptr<T>(), q.ptr<T>(), k.ptr<T>(), v.ptr<T>(), pos, time, slot, cap, nh, nk, dim,
        qn.ptr<T>(), kn.ptr<T>(), eps, gemma, rope.ptr<T>(), fixed_position,
        packed ? packed->row_batch : nullptr, packed ? packed->row_position : nullptr, cache,
        bias.ptr<T>(), in.ld(), q.ld());
}
void Backend::qkv(TensorView in, TensorView q, TensorView k, TensorView v, const int *pos,
                  int batch, int time, int slot, int cap, int nh, int nk, int dim, TensorView qn,
                  TensorView kn, float eps, bool gemma, TensorView rope, int fixed_position,
                  const SequenceLayout *packed, CacheLayout cache, TensorView bias) {
    check_float_storage(in, {qn, kn, rope});
    if ((bias.data && (bias.numel() != size_t((nh + 2 * nk) * dim) || bias.type != in.type)) ||
        q.type != in.type || (k.cols && k.type != in.type) || (v.cols && v.type != in.type) ||
        (in.type != Type::BF16 && in.type != Type::F16 && in.type != Type::F32))
        throw std::invalid_argument("invalid QKV projection dtype/bias");
    if ((cache.block_table || cache.key_pages) &&
        (!cache.valid_paged() || cap <= 0 ||
         cache.block_stride < (cap + cache.page_size - 1) / cache.page_size))
        throw std::invalid_argument("invalid paged KV layout");
    auto launch = [&](auto type, auto mode) {
        launch_qkv<decltype(type), decltype(mode)::value>(
            stream, in, q, k, v, pos, batch, time, slot, cap, nh, nk, dim, qn, kn, eps, gemma, rope,
            fixed_position, packed, cache, bias);
    };
    auto dispatch = [&](auto type) {
        if (cache.block_table || cache.key_pages)
            launch(type, std::integral_constant<int, 1>{});
        else if (cache.ring)
            launch(type, std::integral_constant<int, 2>{});
        else
            launch(type, std::integral_constant<int, 0>{});
    };
    if (in.type == Type::BF16)
        dispatch(bf16{});
    else if (in.type == Type::F16)
        dispatch(__half{});
    else
        dispatch(float{});
}
template <class T, int Mode>
__global__ void attention_kernel(const T *q, const T *k, const T *v, T *out, const int *positions,
                                 int time, int slot, int cap, int nh, int nk, int dim, bool causal,
                                 int window, int fixed_position, CacheLayout cache) {
    extern __shared__ float scores[];
    __shared__ float maxval, denom;
    int row = blockIdx.x, h = blockIdx.y, b = row / time,
        p = (positions ? positions[b] : fixed_position) + row % time, kh = h / (nh / nk);
    int start = positions ? positions[b] : fixed_position;
    if (cache.sequences) {
        const auto &sequence = cache.sequences[b];
        if (!sequence.position) {
            for (int d = threadIdx.x; d < dim; d += blockDim.x)
                out[(size_t(row) * nh + h) * dim + d] = r<T>(0.f);
            return;
        }
        if (!cache.key_pages) {
            k = reinterpret_cast<const T *>(static_cast<const char *>(sequence.keys) +
                                            cache.layer * sequence.layer_bytes);
            v = reinterpret_cast<const T *>(static_cast<const char *>(sequence.values) +
                                            cache.layer * sequence.layer_bytes);
        }
        start = *sequence.position;
        p = start + row % time;
        cache.block_table = sequence.blocks;
        cache.block_stride = 0;
        b = slot = 0;
    }
    int end = causal ? p + 1 : start + time;
    int begin = window > 0 ? max(0, p - window + 1) : 0;
    if (!causal && window > 0) {
        begin = max(0, p - window);
        end = min(end, p + window + 1);
    }
    int count = end - begin, lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if constexpr (Mode == 1) {
        // Table scratch follows the score scratch, sized from the admitted capacity.
        int *page_ids = reinterpret_cast<int *>(scores + cap);
        for (int page = threadIdx.x + (begin >> cache.page_shift);
             page <= (end - 1) >> cache.page_shift; page += blockDim.x)
            page_ids[page] = cache.block_table[(b + slot) * cache.block_stride + page];
        __syncthreads();
        cache.block_table = page_ids;
        cache.block_stride = 0;
    }
    const T *qp = q + (size_t(row) * nh + h) * dim;
    for (int s = warp; s < count; s += blockDim.x / 32) {
        const T *kp =
            cache_pointer<const T, Mode>(k, b + slot, s + begin, cap, nk * dim, cache, false) +
            kh * dim;
        float sum = 0;
        for (int d = lane; d < dim; d += 32)
            sum += f(qp[d]) * f(kp[d]);
        sum = warp_sum(sum);
        if (!lane)
            scores[s] = sum * rsqrtf(float(dim));
    }
    __syncthreads();
    if (!threadIdx.x) {
        float m = -INFINITY;
        for (int s = 0; s < count; s++)
            m = fmaxf(m, scores[s]);
        maxval = m;
    }
    __syncthreads();
    float sum = 0;
    for (int s = threadIdx.x; s < count; s += blockDim.x) {
        float z = expf(scores[s] - maxval);
        scores[s] = z;
        sum += z;
    }
    sum = block_sum(sum);
    if (!threadIdx.x)
        denom = 1.f / sum;
    __syncthreads();
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float acc = 0;
        if constexpr (Mode == 1) {
            // Unroll independent loads while retaining chronological accumulation.
#pragma unroll 4
            for (int s = 0; s < count; ++s)
                acc +=
                    scores[s] * f(cache_pointer<const T, 1>(v, b + slot, begin + s, cap, nk * dim,
                                                            cache, true)[kh * dim + d]);
        } else if constexpr (Mode == 2) {
            for (int s = 0; s < count;) {
                int physical = (begin + s) % cap;
                int span = min(count - s, cap - physical);
                const T *vp = v + ((size_t(b + slot) * cap + physical) * nk + kh) * dim + d;
#pragma unroll 4
                for (int j = 0; j < span; ++j)
                    acc += scores[s + j] * f(vp[size_t(j) * nk * dim]);
                s += span;
            }
        } else {
            // Keep the tail scalar: CUDA 13 otherwise widens shared loads past
            // the final score at odd capacities (e.g. 17), even if unused lanes
            // do not contribute to the result. No padding is part of this layout.
#pragma unroll 1
            for (int s = 0; s < count; s++)
                acc += scores[s] *
                       f(v[(cache_row<0>(b + slot, s + begin, cap, cache) * nk + kh) * dim + d]);
        }
        out[(size_t(row) * nh + h) * dim + d] = r<T>(acc * denom);
    }
}
// Short attention contexts fit in registers: normalize after the
// cooperative QK reduction, avoiding shared-memory softmax reductions.
static constexpr int short_attention_keys = 16;
__global__ void depth_attention_kernel(const bf16 *q, const bf16 *k, const bf16 *v, bf16 *out,
                                       int time, int cap, int nh, int nk, int dim, int position) {
    __shared__ float scores[short_attention_keys];
    int row = blockIdx.x, h = blockIdx.y, b = row / time, p = position + row % time,
        kh = h / (nh / nk);
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const bf16 *query = q + (size_t(row) * nh + h) * dim;
    for (int t = warp; t <= p; t += 4) {
        float dot = 0;
        const bf16 *key = k + ((size_t(b) * cap + t) * nk + kh) * dim;
        for (int d = lane; d < dim; d += 32)
            dot += f(query[d]) * f(key[d]);
        dot = warp_sum(dot);
        if (!lane)
            scores[t] = dot * rsqrtf(float(dim));
    }
    __syncthreads();
    float maximum = -INFINITY;
    for (int t = 0; t <= p; t++)
        maximum = fmaxf(maximum, scores[t]);
    float weights[short_attention_keys], sum = 0;
    for (int t = 0; t <= p; t++) {
        weights[t] = expf(scores[t] - maximum);
        sum += weights[t];
    }
    for (int d = threadIdx.x; d < dim; d += 128) {
        float value = 0;
        for (int t = 0; t <= p; t++)
            value += weights[t] * f(v[((size_t(b) * cap + t) * nk + kh) * dim + d]);
        out[(size_t(row) * nh + h) * dim + d] = r<bf16>(value / sum);
    }
}
void Backend::attention(TensorView q, TensorView k, TensorView v, TensorView out, const int *pos,
                        int batch, int time, int slot, int cap, int nh, int nk, int dim,
                        bool causal, int window, int fixed_position, const SequenceLayout *packed,
                        CacheLayout cache) {
    check_float_storage(q, {k, v, out});
    if ((cache.block_table || cache.key_pages) &&
        (!cache.valid_paged() || cap <= 0 ||
         cache.block_stride < (cap + cache.page_size - 1) / cache.page_size))
        throw std::invalid_argument("invalid paged attention layout");
    if ((cache.block_table || cache.key_pages) && q.type == Type::BF16 && time == 1 &&
        (pos || cache.sequences) && !packed && dim == 128 && causal && !window && !slot) {
        if (size_t(batch) * nh * sizeof(float) > flash_lse_.size())
            throw std::runtime_error("FlashAttention workspace was not prepared");
        decode_attention(stream, q, k, v, out, flash_lse_.ptr<float>(), pos, batch, nh, nk, cache);
        return;
    }
    if (q.type == Type::BF16 && (time >= 64 || packed) && fixed_position == 0 &&
        ((dim == 128 && causal) || (dim == 256 && !causal))) {
        if (size_t(batch) * time * nh * sizeof(float) > flash_lse_.size())
            throw std::runtime_error("FlashAttention workspace bounds");
        prefill_attention(stream, q, k, v, out, flash_lse_.ptr<float>(), batch, time, slot, cap, nh, nk,
                      dim, causal, window, packed, cache);
        return;
    }
    if (packed) throw std::invalid_argument("packed attention requires a supported FlashAttention path");
    dim3 grid(batch * time, nh);
    size_t shared = cap * sizeof(float);
    if (cache.block_table || cache.key_pages)
        shared += size_t((cap + cache.page_size - 1) / cache.page_size) * sizeof(int);
    if (!pos && !packed && q.type == Type::BF16 && causal && !window && slot == 0 &&
        !cache.block_table && !cache.key_pages && !cache.ring && fixed_position >= 0 && time > 0 &&
        fixed_position + time <= std::min(cap, short_attention_keys)) {
        depth_attention_kernel<<<grid, 128, 0, stream>>>(q.ptr<bf16>(), k.ptr<bf16>(),
                                                         v.ptr<bf16>(), out.ptr<bf16>(), time, cap,
                                                         nh, nk, dim, fixed_position);
        return;
    }
    auto launch = [&](auto type, auto mode) {
        using T = decltype(type);
        attention_kernel<T, decltype(mode)::value><<<grid, 128, shared, stream>>>(
            q.ptr<T>(), k.ptr<T>(), v.ptr<T>(), out.ptr<T>(), pos, time, slot, cap, nh, nk, dim,
            causal, window, fixed_position, cache);
    };
    auto dispatch = [&](auto type) {
        if (cache.block_table || cache.key_pages)
            launch(type, std::integral_constant<int, 1>{});
        else if (cache.ring)
            launch(type, std::integral_constant<int, 2>{});
        else
            launch(type, std::integral_constant<int, 0>{});
    };
    dispatch_float_storage(q.type, dispatch);
}
__global__ void increment_kernel(int *p, int n, int v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        p[i] += v;
}
__global__ void increment_sequences_kernel(const CacheSequence *sequences, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && sequences[i].position)
        ++*sequences[i].position;
}
void Backend::increment(int *p, int n, int v, const CacheSequence *sequences) {
    if (sequences)
        increment_sequences_kernel<<<(n + 255) / 256, 256, 0, stream>>>(sequences, n);
    else
        increment_kernel<<<(n + 255) / 256, 256, 0, stream>>>(p, n, v);
}
} // namespace tts
