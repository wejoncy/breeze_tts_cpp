#include "ops/cuda/kernel_initialization.h"
#include "ops/kernels/attention/relative_attention.h"
#include "ops/relative_attention.h"
#include "ops/kernels/common/cuda_math.cuh"
#include "ops/kernels/common/sequence_address.cuh"
#include "ops/cuda/blas_check.h"
#include "runtime/device.h"
#include <cuda_fp16.h>
#include <cub/block/block_reduce.cuh>
#define FLASH_NAMESPACE tts_flash
#include "flash-attention/flash.h"
#include "flash-attention/flash_fwd_kernel.h"

namespace tts {
namespace {
struct Maximum {
    __device__ float operator()(float a, float b) const {
        return fmaxf(a, b);
    }
};
template <class T>
__global__ void prepare_relative_qkv(T *qkv, int stride, const T *bias, int rows, int width,
                                     int dim) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    int cols = 3 * width;
    if (i >= size_t(rows) * cols)
        return;
    int c = i % cols;
    T *p = qkv + (i / cols) * stride + c;
    float v = rounded<T>(f(*p) + (bias ? f(bias[c]) : 0.f));
    *p = r<T>(c < width ? v * rsqrtf(float(dim)) : v);
}
template <class T, bool CacheScores>
__global__ void relative_softmax(const T *qkv, int qkv_stride, const T *relative_k, T *scores,
                                 int score_stride, int time, int rows, int heads, int dim,
                                 int window, bool shared_heads) {
    extern __shared__ float cache[];
    using Reduce = cub::BlockReduce<float, 256>;
    __shared__ typename Reduce::TempStorage temp;
    __shared__ float maximum, inverse_sum;
    int row = blockIdx.x / heads, head = blockIdx.x % heads, qtime = row % time;
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const T *q = qkv + size_t(row) * qkv_stride + head * dim;
    T *s = scores + (size_t(head) * rows + row) * score_stride;
    if constexpr (CacheScores)
        for (int k = threadIdx.x; k < time; k += blockDim.x)
            cache[k] = f(s[k]);
    __syncthreads();
    int first = max(-window, -qtime), last = min(window, time - 1 - qtime);
    size_t rel_base = (shared_heads ? 0 : size_t(head) * (2 * window + 1)) * dim;
    for (int distance = first + warp; distance <= last; distance += 8) {
        float dot = 0;
        const T *rkey = relative_k + rel_base + size_t(distance + window) * dim;
        for (int d = lane; d < dim; d += 32)
            dot += f(q[d]) * f(rkey[d]);
        dot = warp_sum(dot);
        if (!lane) {
            int k = qtime + distance;
            if constexpr (CacheScores)
                cache[k] = rounded<T>(cache[k] + rounded<T>(dot));
            else
                s[k] = r<T>(f(s[k]) + rounded<T>(dot));
        }
    }
    __syncthreads();
    float value = -INFINITY;
    for (int k = threadIdx.x; k < time; k += blockDim.x)
        value = fmaxf(value, CacheScores ? cache[k] : f(s[k]));
    value = Reduce(temp).Reduce(value, Maximum{});
    if (!threadIdx.x)
        maximum = value;
    __syncthreads();
    value = 0;
    for (int k = threadIdx.x; k < time; k += blockDim.x) {
        float p = __expf((CacheScores ? cache[k] : f(s[k])) - maximum);
        if constexpr (CacheScores)
            cache[k] = p;
        value += p;
    }
    value = Reduce(temp).Sum(value);
    if (!threadIdx.x)
        inverse_sum = 1.f / value;
    __syncthreads();
    for (int k = threadIdx.x; k < time; k += blockDim.x)
        s[k] = r<T>((CacheScores ? cache[k] : __expf(f(s[k]) - maximum)) * inverse_sum);
}
template <class T>
__global__ void relative_value(const T *scores, int score_stride, const T *relative_v, T *out,
                               int out_stride, int time, int rows, int heads, int dim, int window,
                               bool shared_heads) {
    int row = blockIdx.x / heads, head = blockIdx.x % heads, qtime = row % time;
    const T *s = scores + (size_t(head) * rows + row) * score_stride;
    size_t rel_base = (shared_heads ? 0 : size_t(head) * (2 * window + 1)) * dim;
    int first = max(-window, -qtime), last = min(window, time - 1 - qtime);
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float dot = 0;
        for (int distance = first; distance <= last; ++distance)
            dot += f(s[qtime + distance]) *
                   f(relative_v[rel_base + size_t(distance + window) * dim + d]);
        T *p = out + size_t(row) * out_stride + head * dim + d;
        *p = r<T>(f(*p) + rounded<T>(dot));
    }
}
struct OutsideParams : tts_flash::Flash_fwd_params {
    int excluded_window;
    const int *sequence_offsets = nullptr;
};
struct ExcludeBand {
    static constexpr bool check_inf = true;
    template <class Traits, class Fragment, class Mma, class Params>
    __device__ static void apply(Fragment &scores, const Mma &mma, const Params &p, int mblock,
                                 int nblock, int, int, int, int query_time, int key_time) {
        auto coords = mma.partition_C(cute::make_identity_tensor(
            cute::Shape<cute::Int<Traits::kBlockM>, cute::Int<Traits::kBlockN>>{}));
        using E = typename Traits::Element;
#pragma unroll
        for (int i = 0; i < cute::size(scores); ++i) {
            int q = mblock * Traits::kBlockM + cute::get<0>(coords(i));
            int k = nblock * Traits::kBlockN + cute::get<1>(coords(i));
            if (q < query_time && k < key_time)
                scores(i) = abs(k - q) <= p.excluded_window ? -INFINITY : float(E(scores(i)));
        }
    }
};
template <class Traits> __global__ void flash_outside(const __grid_constant__ OutsideParams base) {
    int blocks = (base.seqlen_q + 63) / 64;
    int row = blockIdx.x % blocks, batch_head = blockIdx.x / blocks;
    int lane = batch_head % base.b, head = batch_head / base.b;
    auto p = base;
    if (p.sequence_offsets) {
        int first = p.sequence_offsets[lane], length = p.sequence_offsets[lane + 1] - first;
        if (row * 64 >= length || length < 1) return;
        p.q_ptr = static_cast<char *>(p.q_ptr) + size_t(first) * p.q_row_stride * 2;
        p.k_ptr = static_cast<char *>(p.k_ptr) + size_t(first) * p.k_row_stride * 2;
        p.v_ptr = static_cast<char *>(p.v_ptr) + size_t(first) * p.v_row_stride * 2;
        p.o_ptr = static_cast<char *>(p.o_ptr) + size_t(first) * p.o_row_stride * 2;
        p.oaccum_ptr = static_cast<float *>(p.oaccum_ptr) + size_t(first) * p.h * p.d;
        p.softmax_lse_ptr = static_cast<float *>(p.softmax_lse_ptr) + size_t(first) * p.h;
        p.softmax_lseaccum_ptr = p.softmax_lse_ptr;
        p.q_batch_stride = p.k_batch_stride = p.v_batch_stride = p.o_batch_stride = 0;
        p.b = 1; p.seqlen_q = p.seqlen_k = length;
        p.seqlen_q_rounded = p.seqlen_k_rounded = (length + 127) / 128 * 128;
        lane = 0;
    }
    tts_flash::compute_attn_1rowblock_splitkv<Traits, false, false, false, false, true, false, true,
                                              false, OutsideParams, ExcludeBand>(
        p, lane, head, row, 0, 1);
}
// Online merge over a bounded key tile: fixed shared storage for every context/window.
// FA supplies the normalized outside numerator and its log partition function.
template <class T>
__global__ void merge_relative(const T *qkv, int stride, const T *rk, const T *rv,
                               const float *outside, const float *lse, T *out, int out_stride,
                               int rows, int time, int heads, int dim, int window,
                               bool shared_heads, SequenceLayout packed) {
    constexpr int tile = 64;
    using Reduce = cub::BlockReduce<float, 128>;
    __shared__ typename Reduce::TempStorage temp;
    __shared__ float logits[tile], maximum, denominator, rescale;
    int row = blockIdx.x / heads, head = blockIdx.x % heads;
    auto address = sequence_address(packed, row, time);
    int qtime = address.position, batch = address.sequence;
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32, d = threadIdx.x;
    size_t outside_row = size_t(address.offset) * heads + size_t(head) * address.extent + qtime;
    const T *q = qkv + size_t(row) * stride + head * dim;
    size_t rel_base = (shared_heads ? 0 : size_t(head) * (2 * window + 1)) * dim;
    float lse_out = lse[outside_row];
    bool has_outside = lse_out != -INFINITY;
    float main = d < dim && has_outside ? outside[outside_row * dim + d] : 0.f;
    float relative = 0;
    if (!threadIdx.x) {
        maximum = lse_out;
        denominator = has_outside ? 1.f : 0.f;
    }
    __syncthreads();
    int first = max(-window, -qtime), last = min(window, address.valid - 1 - qtime);
    for (int begin = first; begin <= last; begin += tile) {
        int count = min(tile, last - begin + 1);
        for (int i = warp; i < count; i += 4) {
            int distance = begin + i;
            const T *k =
                qkv + size_t(address.offset + qtime + distance) * stride + heads * dim + head * dim;
            const T *rkey = rk + rel_base + size_t(distance + window) * dim;
            float dot = 0, rel_dot = 0;
            for (int c = lane; c < dim; c += 32) {
                dot += f(q[c]) * f(k[c]);
                rel_dot += f(q[c]) * f(rkey[c]);
            }
            dot = warp_sum(dot);
            rel_dot = warp_sum(rel_dot);
            if (!lane)
                logits[i] = rounded<T>(rounded<T>(dot) + rounded<T>(rel_dot));
        }
        __syncthreads();
        float value = threadIdx.x < count ? logits[threadIdx.x] : -INFINITY;
        value = Reduce(temp).Reduce(value, Maximum{});
        if (!threadIdx.x) {
            float next = fmaxf(maximum, value);
            rescale = denominator == 0 ? 0.f : __expf(maximum - next);
            maximum = next;
        }
        __syncthreads();
        value = 0;
        if (threadIdx.x < count) {
            value = __expf(logits[threadIdx.x] - maximum);
            logits[threadIdx.x] = value;
        }
        value = Reduce(temp).Sum(value);
        if (!threadIdx.x)
            denominator = denominator * rescale + value;
        __syncthreads();
        if (d < dim) {
            main *= rescale;
            relative *= rescale;
            for (int i = 0; i < count; ++i) {
                int distance = begin + i;
                size_t offset = size_t(address.offset + qtime + distance) * stride + 2 * heads * dim +
                                head * dim + d;
                main += logits[i] * f(qkv[offset]);
                relative += logits[i] * f(rv[rel_base + size_t(distance + window) * dim + d]);
            }
        }
        __syncthreads();
    }
    if (d < dim)
        out[size_t(row) * out_stride + head * dim + d] =
            r<T>(rounded<T>(main / denominator) + rounded<T>(relative / denominator));
}
template <class Element, int Dim> void launch_outside(cudaStream_t stream, const OutsideParams &p) {
    using Traits = Flash_fwd_kernel_traits<Dim, 64, 64, 4, false, false, Element>;
    constexpr size_t smem =
        Traits::kSmemSize > 64 * Dim * sizeof(float) ? Traits::kSmemSize : 64 * Dim * sizeof(float);
    flash_outside<Traits><<<size_t((p.seqlen_q + 63) / 64) * p.b * p.h, 128, smem, stream>>>(p);
}
void flash_relative(cudaStream_t stream, TensorView qkv, TensorView rk, TensorView rv,
                    TensorView out, void *scratch, int batch, int time, int heads, int window,
                    const SequenceLayout *packed) {
    int rows = packed ? packed->rows : batch * time, dim = out.cols / heads;
    OutsideParams p{};
    p.q_ptr = qkv.data;
    p.k_ptr = qkv.columns(out.cols, out.cols).data;
    p.v_ptr = qkv.columns(2 * out.cols, out.cols).data;
    p.o_ptr = out.data;
    p.oaccum_ptr = scratch;
    auto *lse = reinterpret_cast<float *>(
        static_cast<char *>(scratch) +
        Tensor::aligned(size_t(rows) * heads * dim * sizeof(float)));
    p.softmax_lse_ptr = p.softmax_lseaccum_ptr = lse;
    p.q_batch_stride = p.k_batch_stride = p.v_batch_stride = int64_t(time) * qkv.ld();
    p.q_row_stride = p.k_row_stride = p.v_row_stride = qkv.ld();
    p.q_head_stride = p.k_head_stride = p.v_head_stride = dim;
    p.o_row_stride = out.ld();
    p.o_batch_stride = int64_t(time) * out.ld();
    p.o_head_stride = dim;
    p.b = batch;
    p.h = p.h_k = heads;
    p.h_h_k_ratio = 1;
    p.d = p.d_rounded = dim;
    p.seqlen_q = p.seqlen_k = time;
    p.seqlen_q_rounded = p.seqlen_k_rounded = (time + 127) / 128 * 128;
    p.scale_softmax = 1.f;
    p.scale_softmax_log2 = 1.4426950408889634f;
    p.p_dropout = p.rp_dropout = p.scale_softmax_rp_dropout = 1.f;
    p.p_dropout_in_uint8_t = 255;
    p.is_bf16 = out.type == Type::BF16;
    p.is_seqlens_k_cumulative = true;
    p.window_size_left = p.window_size_right = -1;
    p.excluded_window = window;
    p.sequence_offsets = packed ? packed->cu_seqlens : nullptr;
    auto dispatch = [&](auto value) {
        using E = decltype(value);
        if (dim == 32)
            launch_outside<E, 32>(stream, p);
        else if (dim == 64)
            launch_outside<E, 64>(stream, p);
        else if (dim == 96)
            launch_outside<E, 96>(stream, p);
        else
            launch_outside<E, 128>(stream, p);
    };
    if (p.is_bf16)
        dispatch(cutlass::bfloat16_t{});
    else
        dispatch(cutlass::half_t{});
    auto finish = [&](auto value) {
        using T = decltype(value);
        merge_relative<T><<<rows * heads, 128, 0, stream>>>(
            qkv.ptr<T>(), qkv.ld(), rk.ptr<T>(), rv.ptr<T>(), static_cast<float *>(scratch), lse,
            out.ptr<T>(), out.ld(), rows, time, heads, dim, window, rk.rows == 2 * window + 1,
            packed ? *packed : SequenceLayout{});
    };
    if (out.type == Type::F16)
        finish(__half{});
    else
        finish(bf16{});
}
bool use_flash(Type type, int dim) {
    return (type == Type::F16 || type == Type::BF16) &&
           (dim == 32 || dim == 64 || dim == 96 || dim == 128);
}
int relative_attention_score_stride(int time, Type type) {
    int align = 16 / bytes(type);
    if (time < 1 || time > INT32_MAX - align + 1)
        throw std::invalid_argument("invalid relative attention time");
    return (time + align - 1) / align * align;
}
size_t workspace_rows(int batch, int time, int heads, int dim, Type type) {
    if (batch < 1 || time < 1 || time > INT32_MAX - 127 || heads < 1 || dim < 1 ||
        (type != Type::BF16 && type != Type::F16 && type != Type::F32) ||
        size_t(batch) * heads > SIZE_MAX / size_t(time))
        throw std::invalid_argument("invalid relative attention workspace geometry");
    return size_t(batch) * heads * time;
}
} // namespace
size_t ops::relative_attention_workspace_bytes(int batch, int time, int heads, int dim, Type type) {
    size_t rows = workspace_rows(batch, time, heads, dim, type);
    if (use_flash(type, dim)) {
        if (rows > (SIZE_MAX - 255) / sizeof(float) / (size_t(dim) + 1))
            throw std::overflow_error("relative attention workspace size");
        return Tensor::aligned(rows * dim * sizeof(float)) + rows * sizeof(float);
    }
    size_t stride = relative_attention_score_stride(time, type);
    if (rows > SIZE_MAX / stride / bytes(type))
        throw std::overflow_error("relative attention workspace size");
    return rows * stride * bytes(type);
}
void cuda_relative_attention(cublasHandle_t blas, cudaStream_t stream, TensorView qkv, TensorView bias,
                        TensorView rel_k, TensorView rel_v, TensorView out, void *scratch,
                        size_t scratch_bytes, int batch, int time, int heads, int window,
                        size_t shared_memory_limit, const SequenceLayout *packed) {
    int64_t rows = packed ? packed->rows : int64_t(batch) * time;
    if (batch < 1 || time < 1 || heads < 1 || window < 0 || window > (INT32_MAX - 1) / 2 ||
        out.cols < 1 || out.cols % heads || qkv.rows != rows || out.rows != rows ||
        rows > INT32_MAX / heads || qkv.cols != int64_t(out.cols) * 3 || qkv.type != out.type ||
        (out.type != Type::BF16 && out.type != Type::F16 && out.type != Type::F32) ||
        (bias.data &&
         (bias.numel() != size_t(qkv.cols) || bias.type != qkv.type || !bias.dense())) ||
        rel_k.type != qkv.type || rel_v.type != qkv.type || rel_k.cols != out.cols / heads ||
        rel_v.cols != rel_k.cols || rel_k.rows != rel_v.rows || !rel_k.dense() || !rel_v.dense() ||
        (rel_k.rows != 2 * window + 1 && rel_k.rows != int64_t(heads) * (2 * window + 1)) ||
        !scratch ||
        scratch_bytes <
            ops::relative_attention_workspace_bytes(packed ? 1 : batch, packed ? packed->rows : time,
                                                    heads, out.cols / heads, qkv.type))
        throw std::invalid_argument("invalid relative attention geometry");
    int width = out.cols, dim = width / heads;
    if (packed && (!packed->cu_seqlens || !use_flash(out.type, dim)))
        throw std::invalid_argument("packed relative attention requires supported FlashAttention geometry");
    if (use_flash(out.type, dim) && (reinterpret_cast<uintptr_t>(qkv.data) % 16 || qkv.ld() % 8 ||
                                     reinterpret_cast<uintptr_t>(scratch) % 16))
        throw std::invalid_argument(
            "relative FlashAttention requires 16-byte aligned QKV rows and workspace");
    auto launch_prepare = [&](auto type) {
        using T = decltype(type);
        prepare_relative_qkv<T><<<(size_t(rows) * qkv.cols + 255) / 256, 256, 0, stream>>>(
            qkv.ptr<T>(), qkv.ld(), bias.ptr<T>(), rows, width, dim);
    };
    if (out.type == Type::BF16)
        launch_prepare(bf16{});
    else if (out.type == Type::F16)
        launch_prepare(__half{});
    else
        launch_prepare(float{});
    if (use_flash(out.type, dim)) {
        flash_relative(stream, qkv, rel_k, rel_v, out, scratch, batch, time, heads, window, packed);
        CUDA(cudaGetLastError());
        return;
    }
    TensorView scores(scratch, int(rows * heads), time, qkv.type,
                      relative_attention_score_stride(time, qkv.type));
    auto dt = qkv.type == Type::BF16  ? CUDA_R_16BF
              : qkv.type == Type::F16 ? CUDA_R_16F
                                      : CUDA_R_32F;
    auto compute = qkv.type == Type::F32 ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F;
    float alpha = 1, beta = 0;
    size_t unit = bytes(qkv.type);
    long long qkv_batch = static_cast<long long>(time) * qkv.ld();
    long long score_batch = static_cast<long long>(time) * scores.ld();
    auto ptr = [unit](TensorView v, size_t offset) {
        return static_cast<char *>(v.data) + offset * unit;
    };
    for (int h = 0; h < heads; ++h)
        BLAS(cublasGemmStridedBatchedEx(
            blas, CUBLAS_OP_T, CUBLAS_OP_N, time, time, dim, &alpha, ptr(qkv, width + h * dim), dt,
            qkv.ld(), qkv_batch, ptr(qkv, h * dim), dt, qkv.ld(), qkv_batch, &beta,
            ptr(scores, size_t(h) * batch * score_batch), dt, scores.ld(), score_batch, batch,
            compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    auto launch_softmax = [&](auto type) {
        using T = decltype(type);
        const size_t static_shared = Tensor::aligned(
            sizeof(typename cub::BlockReduce<float, 256>::TempStorage) + 2 * sizeof(float));
        if (size_t(time) * sizeof(float) + static_shared <= shared_memory_limit)
            relative_softmax<T, true><<<rows * heads, 256, size_t(time) * sizeof(float), stream>>>(
                qkv.ptr<T>(), qkv.ld(), rel_k.ptr<T>(), scores.ptr<T>(), scores.ld(), time, rows,
                heads, dim, window, rel_k.rows == 2 * window + 1);
        else
            relative_softmax<T, false><<<rows * heads, 256, 0, stream>>>(
                qkv.ptr<T>(), qkv.ld(), rel_k.ptr<T>(), scores.ptr<T>(), scores.ld(), time, rows,
                heads, dim, window, rel_k.rows == 2 * window + 1);
    };
    if (out.type == Type::BF16)
        launch_softmax(bf16{});
    else if (out.type == Type::F16)
        launch_softmax(__half{});
    else
        launch_softmax(float{});
    for (int h = 0; h < heads; ++h)
        BLAS(cublasGemmStridedBatchedEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, dim, time, time, &alpha,
                                        ptr(qkv, 2 * width + h * dim), dt, qkv.ld(), qkv_batch,
                                        ptr(scores, size_t(h) * batch * score_batch), dt,
                                        scores.ld(), score_batch, &beta, ptr(out, h * dim), dt,
                                        out.ld(), static_cast<long long>(time) * out.ld(), batch,
                                        compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    auto launch_value = [&](auto type) {
        using T = decltype(type);
        relative_value<T><<<rows * heads, 128, 0, stream>>>(
            scores.ptr<T>(), scores.ld(), rel_v.ptr<T>(), out.ptr<T>(), out.ld(), time, rows, heads,
            dim, window, rel_v.rows == 2 * window + 1);
    };
    if (out.type == Type::BF16)
        launch_value(bf16{});
    else if (out.type == Type::F16)
        launch_value(__half{});
    else
        launch_value(float{});
    CUDA(cudaGetLastError());
}
void cuda_kernels::initialize_relative_attention() {
    auto scalar = [](auto value) {
        using T = decltype(value);
        materialize(prepare_relative_qkv<T>);
        materialize(relative_softmax<T, true>);
        materialize(relative_softmax<T, false>);
        materialize(relative_value<T>);
    };
    scalar(bf16{});
    scalar(__half{});
    scalar(float{});
    auto flash = [](auto value) {
        using E = decltype(value);
        materialize(flash_outside<Flash_fwd_kernel_traits<32, 64, 64, 4, false, false, E>>);
        materialize(flash_outside<Flash_fwd_kernel_traits<64, 64, 64, 4, false, false, E>>);
        materialize(flash_outside<Flash_fwd_kernel_traits<96, 64, 64, 4, false, false, E>>);
        materialize(flash_outside<Flash_fwd_kernel_traits<128, 64, 64, 4, false, false, E>>);
    };
    flash(cutlass::bfloat16_t{});
    flash(cutlass::half_t{});
    materialize(merge_relative<bf16>);
    materialize(merge_relative<__half>);
}
} // namespace tts
