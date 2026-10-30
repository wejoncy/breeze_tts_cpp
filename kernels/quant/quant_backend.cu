#include "kernels/quant/quant_backend.h"
#include "backends/cuda/blas_check.h"
#include "kernels/quant/quant_format.cuh"
#include "runtime/prepared_cache.h"
#include "kernels/quant/kernel_selection.h"
#include "kernels/quant/quant_dispatch.h"
#include "kernels/quant/kernel_cost.h"
#include "kernels/quant/quant_norm.h"
#include "kernels/quant/quant_gated.h"
#include <cmath>
#include <cstdint>
#include "support/log.h"
#include <cuda_bf16.h>
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
template <Type Storage>
__global__ void dynamic_token_quantize(const bf16 *x, unsigned char *q, float *scales, int cols,
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
void quantize_rows(Type format, const bf16 *x, unsigned char *q, float *scales, int rows, int cols,
                   int stride, cudaStream_t stream) {
    if (format == Type::I8)
        dynamic_token_quantize<Type::I8><<<rows, 256, 0, stream>>>(x, q, scales, cols, stride);
    else
        dynamic_token_quantize<Type::FP8><<<rows, 256, 0, stream>>>(x, q, scales, cols, stride);
}
struct Plan {
    LinearPath path;
    bool lt_compatible = true;
    int rows = 0, n = 0, k = 0;
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t weight = nullptr, input = nullptr, output = nullptr;
    cublasLtMatmulHeuristicResult_t algorithm{};
    ~Plan() {
        if (operation)
            cublasLtMatmulDescDestroy(operation);
        if (weight)
            cublasLtMatrixLayoutDestroy(weight);
        if (input)
            cublasLtMatrixLayoutDestroy(input);
        if (output)
            cublasLtMatrixLayoutDestroy(output);
    }
};
} // namespace
struct QuantBackend::Impl {
    DeviceContext &memory;
    cudaStream_t stream;
    int device_id;
    Type format;
    size_t shared_bytes = 0;
    ChipModel chip;
    bool hopper = false;
    QuantNorm norm;
    QuantGated gated;
    DeviceArena arena;
    Buffer workspace;
    cublasLtHandle_t handle = nullptr;
    KernelSelector<LinearPath> selector;
    PreparedCache<std::tuple<int, int, int>, Plan> plans;
    Impl(DeviceContext &m, Type f)
        : memory(m), stream(m.stream), device_id(m.device), format(f), norm(m, f), gated(m, f),
          arena(m, size_t(8) << 20), workspace(size_t(8) << 20, &arena) {
        cudaDeviceProp property{};
        CUDA(cudaGetDeviceProperties(&property, device_id));
        chip.sms = property.multiProcessorCount;
        int sm_clock = 0, memory_clock = 0, memory_bus = 0;
        CUDA(cudaDeviceGetAttribute(&sm_clock, cudaDevAttrClockRate, device_id));
        CUDA(cudaDeviceGetAttribute(&memory_clock, cudaDevAttrMemoryClockRate, device_id));
        CUDA(cudaDeviceGetAttribute(&memory_bus, cudaDevAttrGlobalMemoryBusWidth, device_id));
        chip.memory_bytes_per_us = 2. * memory_clock / 1000 * memory_bus / 8;
        // Dense 8-bit Tensor Core throughput: SM80 has twice GA10x's rate;
        // SM90 doubles SM80. Scalar FP8 also pays conversion instructions.
        int tensor_ops = property.major >= 9 ? 8192 : property.minor == 0 ? 4096 : 2048;
        int scalar_lanes = property.major == 8 && property.minor == 0 ? 64 : 128;
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
        shared_bytes = property.sharedMemPerBlock;
        hopper = property.major == 9 && property.minor == 0;
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
#else
        BLAS(cublasLtCreate(&handle));
#endif
    }
    void run(Plan &p, void *x, const float *xs, void *w, const float *ws, void *y,
             int actual_rows = 0) {
        int m = actual_rows ? actual_rows : p.rows, n = p.n, k = p.k;
        if (p.path.kind != LinearKind::Lt) {
            launch_quant_linear(p.path, {m, n, k}, x, xs, w, ws, y, workspace.data(),
                                workspace.size(), stream, format);
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
        float alpha = 1, beta = 0;
        BLAS(cublasLtMatmul(handle, p.operation, &alpha, w, p.weight, x, p.input, &beta, y,
                            p.output, y, p.output, &p.algorithm.algo, workspace.data(),
                            workspace.size(), stream));
    }
    std::unique_ptr<Plan> make_plan(int rows, int n, int k) {
        if (rows < 1 || n < 1 || k < 1 || k % 16 || n % 16)
            throw std::invalid_argument("quantized GEMM unsupported shape");
        auto p = std::make_unique<Plan>();
        p->rows = rows;
        p->n = n;
        p->k = k;
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
        BLAS(cublasLtMatrixLayoutCreate(&p->weight, CUDA_R_8F_E4M3, k, n, k));
        BLAS(cublasLtMatrixLayoutCreate(&p->input, CUDA_R_8F_E4M3, k, rows, k));
        BLAS(cublasLtMatrixLayoutCreate(&p->output, CUDA_R_16BF, n, rows, n));
        cublasLtMatmulPreference_t preference;
        BLAS(cublasLtMatmulPreferenceCreate(&preference));
        size_t bytes = workspace.size();
        auto status = cublasLtMatmulPreferenceSetAttribute(
            preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &bytes, sizeof(bytes));
        int count = 0;
        if (status == CUBLAS_STATUS_SUCCESS)
            status =
                cublasLtMatmulAlgoGetHeuristic(handle, p->operation, p->weight, p->input, p->output,
                                               p->output, preference, 1, &p->algorithm, &count);
        cublasLtMatmulPreferenceDestroy(preference);
        if (status != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("FP8 GEMM preparation rows=" + std::to_string(rows) +
                                     " out=" + std::to_string(n) + " in=" + std::to_string(k) +
                                     ": cuBLAS status " + std::to_string(int(status)));
        if (!count)
            throw std::runtime_error("cuBLASLt has no FP8 outer-scale GEMM for this shape");
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
        // Load the library's FP8 kernels before serving. One WGMMA-sized
        // GEMM primes the shared module; no model shapes or timing search.
        constexpr int extent = 128;
        constexpr size_t elements = size_t(extent) * extent;
        Buffer scratch(4 * elements + 2 * extent * sizeof(float), &memory);
        auto x = scratch.view(extent, extent, format);
        auto w = scratch.view(extent, extent, format, elements);
        auto y = scratch.view(extent, extent, Type::BF16, 2 * elements);
        auto sx = scratch.view(extent, 1, Type::F32, 4 * elements);
        auto sw = scratch.view(extent, 1, Type::F32, 4 * elements + extent * sizeof(float));
        auto p = make_plan(extent, extent, extent);
        CUDA(cudaMemsetAsync(scratch.data(), 0, scratch.size(), stream));
        run(*p, x.data, sx.ptr<float>(), w.data, sw.ptr<float>(), y.data);
        CUDA(cudaStreamSynchronize(stream));
    }
    ~Impl() {
        if (handle)
            cublasLtDestroy(handle);
    }
};
QuantBackend::QuantBackend(DeviceContext &memory, Type format)
    : impl(std::make_unique<Impl>(memory, format)) {
    impl->initialize_library();
}
QuantBackend::~QuantBackend() = default;
Type QuantBackend::type() const {
    return impl->format;
}
void QuantBackend::seal() {
    impl->plans.seal();
}
void QuantBackend::begin_admission() {
    impl->plans.begin_admission();
}
QuantCacheStats QuantBackend::cache_stats() const {
    return {impl->selector.stats(), impl->plans.size()};
}
void QuantBackend::prepare(int rows, int n, int k) {
    rows = row_capacity(rows);
    impl->plans.prepare(std::make_tuple(rows, n, k), [&] {
        auto p = impl->make_plan(rows, n, k);
        p->path = impl->selector.select({rows, n, k}, [&](KernelProblem problem) {
            auto path = select_linear(
                problem, {impl->hopper, impl->shared_bytes, impl->workspace.size()}, impl->format,
                impl->chip, p->lt_compatible, p->algorithm.wavesCount);
            TTS_LOG(TTS_LOG_DEBUG, "quant.gemm", "{} planned m={} n={} k={} path={}",
                    quantized_name(impl->format), problem.rows, n, k, path_name(path));
            return path;
        });
        return p;
    });
}
void QuantBackend::quantize(TensorView source, RowScaledView destination) {
    destination.validate();
    if (destination.values.type != impl->format || source.type != Type::BF16 ||
        source.rows != destination.values.rows || source.cols != destination.values.cols)
        throw std::invalid_argument("quantized quantization shape/type mismatch");
    quantize_rows(impl->format, source.ptr<bf16>(), destination.values.ptr<unsigned char>(),
                  destination.scales.ptr<float>(), source.rows, source.cols, source.ld(),
                  impl->stream);
}
void QuantBackend::linear(RowScaledView x, RowScaledView w, TensorView y) {
    x.validate();
    w.validate();
    if (x.values.type != impl->format || w.values.type != impl->format ||
        x.values.cols != w.values.cols || x.values.rows != y.rows || w.values.rows != y.cols ||
        y.type != Type::BF16 || !y.dense())
        throw std::invalid_argument("quantized linear dimensions/type mismatch");
    auto *plan = impl->plans.find(std::make_tuple(row_capacity(y.rows), y.cols, x.values.cols));
    if (!plan)
        throw std::logic_error("quantized GEMM shape was not prepared before execution");
    impl->run(*plan, x.values.data, x.scales.ptr<float>(), w.values.data, w.scales.ptr<float>(),
              y.data, y.rows);
}
void QuantBackend::norm(TensorView x, TensorView update, TensorView w, RowScaledView out, float eps,
                        bool gemma, TensorView update_scale) {
    impl->norm.run(x, update, w, out, eps, gemma, update_scale);
}
void QuantBackend::gated(TensorView input, RowScaledView out, int kind) {
    impl->gated.run(input, out, kind);
}
} // namespace tts
