#include "ops/cuda/quantization.h"
#include "ops/cuda/lt_plan.h"
#include "ops/kernels/quant/quant_device.cuh"
#include "runtime/prepared_cache.h"
#include "ops/kernels/quant/kernel_selection.h"
#include "ops/kernels/quant/quant_dispatch.h"
#include "ops/kernels/quant/kernel_cost.h"
#include "ops/kernels/common/chip_execution.h"
#include "ops/kernels/sequence/norm_projection.h"
#include "ops/kernels/quant/quant_norm.h"
#include "ops/kernels/quant/quant_gated.h"
#include <cmath>
#include <cstdint>
#include "support/log.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <array>
#include <cublasLt.h>
#include <tuple>

namespace tts {
using bf16 = __nv_bfloat16;
namespace {
template <int Threads = 256> __device__ float reduce_max(float value) {
    __shared__ float partial[Threads / 32];
    for (int d = 16; d; d /= 2)
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, d));
    if (!(threadIdx.x % 32))
        partial[threadIdx.x / 32] = value;
    __syncthreads();
    int lane = threadIdx.x % 32;
    float result = lane < Threads / 32 ? partial[lane] : 0.f;
    for (int d = 16; d; d /= 2)
        result = fmaxf(result, __shfl_down_sync(0xffffffff, result, d));
    return __shfl_sync(0xffffffff, result, 0);
}
template <Type Storage, class T>
__global__ void dynamic_token_quantize(const T *x, unsigned char *q, float *scales, int cols,
                                       int stride) {
    int row = blockIdx.x;
    float amax = 0;
    for (int c = threadIdx.x; c < cols; c += 256)
        amax = fmaxf(amax, fabsf(float(x[size_t(row) * stride + c])));
    float scale = quant_device::scale<Storage>(reduce_max(amax)), inv = 1.f / scale;
    if (!threadIdx.x)
        scales[row] = scale;
    for (int c = threadIdx.x; c < cols; c += 256)
        q[size_t(row) * cols + c] =
            quant_device::Format<Storage>::encode(float(x[size_t(row) * stride + c]) * inv);
}
// Cache sixteen 16-bit values per thread; wider or unaligned rows keep the
// scalar path. Selection depends only on layout and register-tile capacity.
constexpr int token_quantize_threads = 256;
constexpr int token_quantize_values = 16;
template <Type Format, class T>
__global__ void cached_token_quantize(const T *x, unsigned char *q, float *scales,
                                      int cols, int stride) {
    using Pack = quant_device::ValuePack<T, 4>;
    using Pair = typename quant_device::Packed16<T>::Pair;
    int row = blockIdx.x;
    x += size_t(row) * stride;
    q += size_t(row) * cols;
    constexpr int packets = token_quantize_values / 4;
    Pack values[packets];
    float amax = 0;
#pragma unroll
    for (int i = 0; i < packets; ++i) {
        int c = threadIdx.x * 4 + i * token_quantize_threads * 4;
        Pack value{};
        if (c < cols)
            value = *reinterpret_cast<const Pack *>(x + c);
        values[i] = value;
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            float2 v = quant_device::Packed16<T>::unpack(
                *reinterpret_cast<const Pair *>(value.v + p * 2));
            amax = fmaxf(amax, fmaxf(fabsf(v.x), fabsf(v.y)));
        }
    }
    float scale = quant_device::scale<Format>(quant_device::maximum<token_quantize_threads>(amax));
    float inverse = 1.f / scale;
    if (!threadIdx.x)
        scales[row] = scale;
#pragma unroll
    for (int i = 0; i < packets; ++i) {
        int c = threadIdx.x * 4 + i * token_quantize_threads * 4;
        if (c < cols) {
            Pack value = values[i];
            quant_device::BytePack<4> encoded;
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                float2 v = quant_device::Packed16<T>::unpack(
                    *reinterpret_cast<const Pair *>(value.v + p * 2));
                encoded.v[p] = quant_device::Format<Format>::pair(
                    make_float2(v.x * inverse, v.y * inverse));
            }
            *reinterpret_cast<quant_device::BytePack<4> *>(q + c) = encoded;
        }
    }
}
template <class T>
void quantize_rows(Type format, const T *x, unsigned char *q, float *scales, int rows, int cols,
                   int stride, cudaStream_t stream) {
    using Pack = quant_device::ValuePack<T, 4>;
    bool cached = cols <= token_quantize_threads * token_quantize_values &&
                  cols % 4 == 0 && stride % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(x) % alignof(Pack) == 0 &&
                  reinterpret_cast<uintptr_t>(q) % alignof(quant_device::BytePack<4>) == 0;
    if (format == Type::I8) {
        if (cached)
            cached_token_quantize<Type::I8><<<rows, token_quantize_threads, 0, stream>>>(
                x, q, scales, cols, stride);
        else
            dynamic_token_quantize<Type::I8, T><<<rows, 256, 0, stream>>>(x, q, scales, cols, stride);
    } else {
        if (cached)
            cached_token_quantize<Type::FP8><<<rows, token_quantize_threads, 0, stream>>>(
                x, q, scales, cols, stride);
        else
            dynamic_token_quantize<Type::FP8, T><<<rows, 256, 0, stream>>>(x, q, scales, cols, stride);
    }
}
struct Plan : LtPlan {
    LinearPath path;
    bool lt_compatible = true, finish_available = false;
    int rows = 0, n = 0, k = 0;
    Type output_type = Type::BF16;
};
using PlanKey = std::tuple<int, int, int, Type>;
using FinishKey = std::tuple<PlanKey, ops::Activation, uintptr_t>;
bool simple_finish(const ops::Epilogue &e) {
    return (e.activation == ops::Activation::Relu || e.activation == ops::Activation::GeluTanh) &&
           !e.residual.data && !e.gate.data && !e.condition.data && !e.raw.data &&
           !e.residual_rows && !e.valid_rows_per_sequence && !e.packed &&
           !e.condition_sources && !e.output_rows && e.input_repeat == 1 &&
           e.value_scale == 1 && e.output_scale == 1;
}
void check_finish_bias(const ops::Epilogue &e, int channels, Type type) {
    if (e.bias.data && (e.bias.type != type || !e.bias.dense() ||
                        e.bias.numel() != size_t(channels)))
        throw std::invalid_argument("quantized linear finish bias mismatch");
}
FinishKey finish_key(int rows, int n, int k, Type type, const ops::Epilogue &e) {
    auto address = reinterpret_cast<uintptr_t>(e.bias.data);
    auto alignment = std::min<uintptr_t>(256, address & (~address + 1));
    return {{row_capacity(rows), n, k, type}, e.activation, alignment};
}
void check_linear(Type format, RowScaledView x, RowScaledView w, TensorView y) {
    x.validate();
    w.validate();
    if (x.values.type != format || w.values.type != format ||
        x.values.cols != w.values.cols || x.values.rows != y.rows || w.values.rows != y.cols ||
        (y.type != Type::BF16 && y.type != Type::F16) || !y.dense())
        throw std::invalid_argument("quantized linear dimensions/type mismatch");
}
LinearFinishWork finish_work(const ops::Epilogue &finish) {
    return {(finish.activation == ops::Activation::GeluTanh ? 9. : 1.) +
            1 + (finish.bias.data ? 4 : 0)};
}
} // namespace
struct QuantBackend::Impl {
    DeviceContext &memory;
    cudaStream_t stream;
    int device_id;
    Type format;
    size_t shared_bytes = 0;
    ChipModel chip;
    bool hopper = false;
    int normalized_capacity = 0;
    QuantNorm norm;
    QuantGated gated;
    Buffer &workspace;
    cublasLtHandle_t handle = nullptr;
    // Library compatibility/algorithm costs can differ by destination dtype.
    std::array<KernelSelector<LinearPath>, 2> selectors;
    PreparedCache<PlanKey, Plan> plans;
    PreparedCache<FinishKey, Plan> finishes;
    PreparedCache<std::tuple<LinearKind, int, int, Type>, KernelResources> finish_resources;
    static size_t output_index(Type type) {
        if (type != Type::BF16 && type != Type::F16)
            throw std::invalid_argument("quantized GEMM output must be F16 or BF16");
        return type == Type::F16 ? 1 : 0;
    }
    Impl(DeviceContext &m, cublasLtHandle_t lt, Buffer &scratch, Type f)
        : memory(m), stream(m.stream), device_id(m.device), format(f), norm(m, f), gated(m, f),
          workspace(scratch), handle(lt) {
        cudaDeviceProp property{};
        CUDA(cudaGetDeviceProperties(&property, device_id));
        chip.sms = property.multiProcessorCount;
        normalized_capacity = ops::normalized_quantized_linear_capacity(f, chip.sms);
        int sm_clock = 0, memory_clock = 0, memory_bus = 0;
        CUDA(cudaDeviceGetAttribute(&sm_clock, cudaDevAttrClockRate, device_id));
        CUDA(cudaDeviceGetAttribute(&memory_clock, cudaDevAttrMemoryClockRate, device_id));
        CUDA(cudaDeviceGetAttribute(&memory_bus, cudaDevAttrGlobalMemoryBusWidth, device_id));
        chip.memory_bytes_per_us = 2. * memory_clock / 1000 * memory_bus / 8;
        // Dense 8-bit Tensor Core throughput: SM80 has twice GA10x's rate;
        // SM90 doubles SM80. Scalar FP8 also pays conversion instructions.
        int tensor_ops = property.major >= 9 ? 8192 : property.minor == 0 ? 4096 : 2048;
        int scalar_lanes = fp32_lanes_per_sm(property.major, property.minor);
        chip.tensor_ops_per_us = double(tensor_ops) * sm_clock / 1000;
        chip.warp_tensor_ops_per_us =
            double(property.major >= 9 ? 4096 : tensor_ops) * sm_clock / 1000;
        chip.warp_instructions_per_us = 4. * sm_clock / 1000;
        // Native packed INT8 / FP32 arithmetic rates. The kernel cost policy
        // accounts for FP8 decoding and weight reuse within a token tile.
        chip.scalar_ops_per_us = (f == Type::I8 ? 512. : scalar_lanes * 2.) * sm_clock / 1000;
        chip.cache_bytes = property.l2CacheSize;
        // Approximate aggregate cache-fabric rate, normalized per SM. This is
        // a chip estimate checked with profiler traffic, not a shape lookup.
        chip.cache_bytes_per_us = 64. * chip.sms * sm_clock / 1000;
        // Keep the CUDA driver reservation separate from kernel-owned shared.
        shared_bytes = property.sharedMemPerBlock > property.reservedSharedMemPerBlock
                           ? property.sharedMemPerBlock - property.reservedSharedMemPerBlock
                           : 0;
#if TTS_HAVE_HOPPER_KERNELS
        hopper = property.major == 9 && property.minor == 0;
#endif
        if (format != Type::FP8 && format != Type::I8)
            throw std::invalid_argument("unsupported quantized storage type");
        if (format == Type::I8) {
            if (property.major < 8)
                throw std::runtime_error("INT8 execution requires SM80+");
            return;
        }
        if (property.major < 9)
            throw std::runtime_error("FP8 execution requires SM90+");
#if CUBLAS_VERSION < 130000
        throw std::runtime_error("FP8 outer-vector scaling requires CUDA 13 in this build");
#endif
    }
    void run(Plan &p, void *x, const float *xs, void *w, const float *ws, void *y,
             int actual_rows = 0, const ops::Epilogue *finish = nullptr) {
        int m = actual_rows ? actual_rows : p.rows, n = p.n, k = p.k;
        if (p.path.kind != LinearKind::Lt) {
            launch_quant_linear(p.path, {m, n, k}, x, xs, w, ws, y, workspace.data(),
                                workspace.size(), stream, format, p.output_type, finish);
            return;
        }
        uint64_t columns = m;
        BLAS(cublasLtMatrixLayoutSetAttribute(p.input, CUBLASLT_MATRIX_LAYOUT_COLS, &columns,
                                              sizeof(columns)));
        BLAS(cublasLtMatrixLayoutSetAttribute(p.output, CUBLASLT_MATRIX_LAYOUT_COLS, &columns,
                                              sizeof(columns)));
        BLAS(cublasLtMatmulDescSetAttribute(p.operation, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &ws,
                                            sizeof(ws)));
        BLAS(cublasLtMatmulDescSetAttribute(p.operation, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &xs,
                                            sizeof(xs)));
        if (finish && finish->bias.data)
            BLAS(cublasLtMatmulDescSetAttribute(p.operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                &finish->bias.data, sizeof(finish->bias.data)));
        float alpha = 1, beta = 0;
        BLAS(cublasLtMatmul(handle, p.operation, &alpha, w, p.weight, x, p.input, &beta, y,
                            p.output, y, p.output, &p.algorithm.algo, workspace.data(),
                            workspace.size(), stream));
    }
    std::unique_ptr<Plan> make_plan(int rows, int n, int k, Type output_type,
                                    const ops::Epilogue *finish = nullptr) {
        output_index(output_type);
        if (rows < 1 || n < 1 || k < 1 || k % 16 || n % 16)
            throw std::invalid_argument("quantized GEMM unsupported shape");
        auto p = std::make_unique<Plan>();
        p->rows = rows;
        p->n = n;
        p->k = k;
        p->output_type = output_type;
        if (format == Type::I8) {
            // Signed byte products accumulate exactly in INT32, including split-K.
            if (k > INT32_MAX / (128 * 128))
                throw std::invalid_argument("INT8 reduction can overflow INT32");
            return p;
        }
#if CUBLAS_VERSION >= 130000
        BLAS(cublasLtMatmulDescCreate(&p->operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        int8_t fast_accum = 0;
        BLAS(cublasLtMatmulDescSetAttribute(p->operation, CUBLASLT_MATMUL_DESC_FAST_ACCUM,
                                            &fast_accum, sizeof(fast_accum)));
        auto transpose = CUBLAS_OP_T;
        BLAS(cublasLtMatmulDescSetAttribute(p->operation, CUBLASLT_MATMUL_DESC_TRANSA, &transpose,
                                            sizeof(transpose)));
        auto mode = CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
        BLAS(cublasLtMatmulDescSetAttribute(p->operation, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode,
                                            sizeof(mode)));
        BLAS(cublasLtMatmulDescSetAttribute(p->operation, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode,
                                            sizeof(mode)));
        // Outer-vector heuristics require non-null device scale pointers. Planning
        // does not execute the GEMM; bind aligned storage, then replace both pointers
        // with the actual row scales in linear() before every launch/capture.
        const void *planning_scale = workspace.data();
        BLAS(cublasLtMatmulDescSetAttribute(p->operation, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                            &planning_scale, sizeof(planning_scale)));
        BLAS(cublasLtMatmulDescSetAttribute(p->operation, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                            &planning_scale, sizeof(planning_scale)));
        if (finish) {
            auto epilogue = finish->activation == ops::Activation::GeluTanh
                ? (finish->bias.data ? CUBLASLT_EPILOGUE_GELU_BIAS : CUBLASLT_EPILOGUE_GELU)
                : (finish->bias.data ? CUBLASLT_EPILOGUE_RELU_BIAS : CUBLASLT_EPILOGUE_RELU);
            BLAS(cublasLtMatmulDescSetAttribute(p->operation, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                &epilogue, sizeof(epilogue)));
            if (finish->bias.data) {
                auto bias_type = output_type == Type::F16 ? CUDA_R_16F : CUDA_R_16BF;
                BLAS(cublasLtMatmulDescSetAttribute(p->operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                    &finish->bias.data, sizeof(finish->bias.data)));
                BLAS(cublasLtMatmulDescSetAttribute(p->operation, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                                    &bias_type, sizeof(bias_type)));
            }
        }
        BLAS(cublasLtMatrixLayoutCreate(&p->weight, CUDA_R_8F_E4M3, k, n, k));
        BLAS(cublasLtMatrixLayoutCreate(&p->input, CUDA_R_8F_E4M3, k, rows, k));
        BLAS(cublasLtMatrixLayoutCreate(
            &p->output, output_type == Type::F16 ? CUDA_R_16F : CUDA_R_16BF, n, rows, n));
        if (!p->select(handle, workspace.size())) {
            p->lt_compatible = false;
            return p;
        }
        // A selected algorithm must support every actual M in this capacity
        // class. This is a host-only initialization check, never a serving query.
        int lower = rows;
        while (lower > 1 && row_capacity(lower - 1) == rows)
            --lower;
        for (int actual = lower; actual <= rows; ++actual) {
            uint64_t columns = actual;
            BLAS(cublasLtMatrixLayoutSetAttribute(p->input, CUBLASLT_MATRIX_LAYOUT_COLS, &columns,
                                                  sizeof(columns)));
            BLAS(cublasLtMatrixLayoutSetAttribute(p->output, CUBLASLT_MATRIX_LAYOUT_COLS, &columns,
                                                  sizeof(columns)));
            cublasLtMatmulHeuristicResult_t checked{};
            auto status =
                cublasLtMatmulAlgoCheck(handle, p->operation, p->weight, p->input, p->output,
                                        p->output, &p->algorithm.algo, &checked);
            if (status != CUBLAS_STATUS_SUCCESS || checked.state != CUBLAS_STATUS_SUCCESS ||
                checked.workspaceSize > workspace.size())
                p->lt_compatible = false;
        }
        return p;
#else
        throw std::runtime_error("FP8 execution requires CUDA 13");
#endif
    }
    void initialize_library() {
        if (format != Type::FP8)
            return;
        // Prime supported FP8 destination types before serving; fixed WGMMA
        // geometry, with no model shapes or timing search.
        constexpr int extent = 128;
        constexpr size_t elements = size_t(extent) * extent;
        Buffer scratch(4 * elements + 2 * extent * sizeof(float), &memory);
        auto x = scratch.view(extent, extent, format);
        auto w = scratch.view(extent, extent, format, elements);
        auto sx = scratch.view(extent, 1, Type::F32, 4 * elements);
        auto sw = scratch.view(extent, 1, Type::F32, 4 * elements + extent * sizeof(float));
        CUDA(cudaMemsetAsync(scratch.data(), 0, scratch.size(), stream));
        for (auto type : {Type::BF16, Type::F16}) {
            auto p = make_plan(extent, extent, extent, type);
            if (p->lt_compatible) {
                auto y = scratch.view(extent, extent, type, 2 * elements);
                run(*p, x.data, sx.ptr<float>(), w.data, sw.ptr<float>(), y.data);
            }
        }
        CUDA(cudaStreamSynchronize(stream));
    }
};
QuantBackend::QuantBackend(DeviceContext &memory, cublasLtHandle_t handle, Buffer &workspace,
                           Type format)
    : impl(std::make_unique<Impl>(memory, handle, workspace, format)) {
    impl->initialize_library();
}
QuantBackend::~QuantBackend() = default;
Type QuantBackend::type() const {
    return impl->format;
}
void QuantBackend::seal() {
    impl->plans.seal();
    impl->finishes.seal();
    impl->finish_resources.seal();
}
QuantCacheStats QuantBackend::cache_stats() const {
    QuantCacheStats result{};
    result.prepared_plans = impl->plans.size() + impl->finishes.size();
    for (const auto &selector : impl->selectors) {
        auto s = selector.stats();
        result.selection.entries += s.entries;
        result.selection.hits += s.hits;
        result.selection.searches += s.searches;
    }
    return result;
}
void QuantBackend::prepare(int rows, int n, int k, Type output_type) {
    auto &selector = impl->selectors[Impl::output_index(output_type)];
    rows = row_capacity(rows);
    impl->plans.prepare(std::make_tuple(rows, n, k, output_type), [&] {
        auto p = impl->make_plan(rows, n, k, output_type);
        p->path = selector.select({rows, n, k}, [&](KernelProblem problem) {
            auto path =
                select_linear(problem, {impl->hopper, impl->shared_bytes, impl->workspace.size()},
                              impl->format, impl->chip, p->lt_compatible, p->work);
            TTS_LOG(TTS_LOG_DEBUG, "quant.gemm", "{} planned m={} n={} k={} output={} path={}",
                    quantized_name(impl->format), problem.rows, n, k,
                    output_type == Type::F16 ? "f16" : "bf16", path_name(path));
            return path;
        });
        return p;
    });
}
void QuantBackend::prepare_finish(int rows, int n, int k, Type output_type,
                                  const ops::Epilogue &finish) {
    if (!simple_finish(finish) || (impl->format == Type::I8 && !impl->hopper)) return;
    check_finish_bias(finish, n, output_type);
    // The native column broadcast uses up to one 128-bit vector load.
    if (impl->format == Type::I8 && reinterpret_cast<uintptr_t>(finish.bias.data) % 16) return;
    rows = row_capacity(rows);
    auto *ordinary = impl->plans.find({rows, n, k, output_type});
    if (!ordinary) throw std::logic_error("quantized finish requires a prepared projection");
    if (impl->format == Type::FP8 && ordinary->path.kind != LinearKind::Lt) return;
    impl->finishes.prepare(finish_key(rows, n, k, output_type, finish), [&] {
        auto p = impl->make_plan(rows, n, k, output_type,
                                 impl->format == Type::FP8 ? &finish : nullptr);
        KernelProblem problem{rows, n, k};
        auto work = finish_work(finish);
        double best = linear_pipeline_cost(problem, ordinary->path, impl->format, impl->chip,
                                            work, false, ordinary->work);
        int best_registers = std::numeric_limits<int>::max();
        auto consider = [&](LinearPath path, LibraryWork library, KernelResources resources) {
            double cost = linear_pipeline_cost(problem, path, impl->format, impl->chip,
                                                work, true, library, resources);
            if (cost < best || (p->finish_available && cost == best &&
                               resources.registers < best_registers)) {
                p->path = path; p->finish_available = true;
                best = cost; best_registers = resources.registers;
            }
        };
        if (impl->format == Type::FP8) {
            if (p->lt_compatible) consider({}, p->work, {});
        } else {
            auto paths = quant_linear_candidates(problem,
                {impl->hopper, impl->shared_bytes, impl->workspace.size()}, impl->format);
            for (auto path : paths) {
                if (path.kind != LinearKind::Hopper || path.splits != 1) continue;
                auto &resources = impl->finish_resources.prepare(
                    {path.kind, path.channels, path.tokens, output_type}, [&] {
                        return std::make_unique<KernelResources>(
                            quant_linear_finish_resources(path, impl->format, output_type));
                    });
                consider(path, {}, resources);
            }
        }
        if (p->finish_available)
            TTS_LOG(TTS_LOG_DEBUG, "quant.finish", "{} planned m={} n={} k={} output={} path={} estimated_pipeline_us={}",
                    quantized_name(impl->format), rows, n, k,
                    output_type == Type::F16 ? "f16" : "bf16", path_name(p->path), best);
        return p;
    });
}
void QuantBackend::quantize(TensorView source, RowScaledView destination) {
    destination.validate();
    if (destination.values.type != impl->format ||
        (source.type != Type::BF16 && source.type != Type::F16) ||
        source.rows != destination.values.rows || source.cols != destination.values.cols)
        throw std::invalid_argument("quantized quantization shape/type mismatch");
    if (source.type == Type::F16)
        quantize_rows(impl->format, source.ptr<__half>(), destination.values.ptr<unsigned char>(),
                      destination.scales.ptr<float>(), source.rows, source.cols, source.ld(),
                      impl->stream);
    else
        quantize_rows(impl->format, source.ptr<bf16>(), destination.values.ptr<unsigned char>(),
                      destination.scales.ptr<float>(), source.rows, source.cols, source.ld(),
                      impl->stream);
}
void QuantBackend::linear(RowScaledView x, RowScaledView w, TensorView y) {
    check_linear(impl->format, x, w, y);
    auto *plan =
        impl->plans.find(std::make_tuple(row_capacity(y.rows), y.cols, x.values.cols, y.type));
    if (!plan)
        throw std::logic_error("quantized GEMM shape was not prepared before execution");
    impl->run(*plan, x.values.data, x.scales.ptr<float>(), w.values.data, w.scales.ptr<float>(),
              y.data, y.rows);
}
bool QuantBackend::linear_finish(RowScaledView x, RowScaledView w, TensorView y,
                                 const ops::Epilogue &finish) {
    if (!simple_finish(finish)) return false;
    check_linear(impl->format, x, w, y);
    check_finish_bias(finish, y.cols, y.type);
    auto *plan = impl->finishes.find(finish_key(y.rows, y.cols, x.values.cols, y.type, finish));
    if (!plan || !plan->finish_available) return false;
    impl->run(*plan, x.values.data, x.scales.ptr<float>(), w.values.data, w.scales.ptr<float>(),
               y.data, y.rows, &finish);
    return true;
}
bool QuantBackend::normalized_linear(TensorView input, TensorView normalized,
                                      TensorView norm_weight, TensorView norm_bias, float epsilon,
                                      const ops::NormFusion &fusion, RowScaledView weight,
                                      TensorView output, const ops::Epilogue &epilogue,
                                      RowScaledView encoded, const QkvProjectionOutput *writeback) {
    if (epilogue.packed || epilogue.condition_sources || epilogue.output_rows) return false;
    auto *plan = impl->plans.find(std::make_tuple(
        row_capacity(input.rows), weight.values.rows, input.cols, output.type));
    if (!plan || plan->path.kind != LinearKind::Gemv ||
        weight.values.type != impl->format)
        return false;
    return ops::normalized_quantized_linear(impl->stream, input, normalized, norm_weight,
        norm_bias, epsilon, fusion, weight, output, epilogue,
        impl->normalized_capacity, encoded, writeback);
}
void QuantBackend::norm(TensorView x, TensorView update, TensorView w, RowScaledView out, float eps,
                        bool gemma, TensorView update_scale) {
    impl->norm.run(x, update, w, out, eps, gemma, update_scale);
}
void QuantBackend::gated(TensorView input, RowScaledView out, int kind) {
    impl->gated.run(input, out, kind);
}
} // namespace tts
