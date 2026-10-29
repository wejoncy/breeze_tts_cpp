#include "quant_backend.h"
#include "blas_check.h"
#include "quant_kernel_catalog.h"
#include "quant_format.cuh"
#include "prepared_cache.h"
#include "kernel_selection.h"
#include "quant_dispatch.h"
#include "kernel_timing.h"
#include "quant_norm.h"
#include "quant_gated.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <functional>
#include "log.h"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublasLt.h>
#include <cfloat>
#include <map>
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
__global__ void initialize_quant_tuning(unsigned char *p, size_t count) {
    size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (i < count) {
        unsigned int state = unsigned(i) * 1664525u + 1013904223u;
        p[i] = ((state ^ (state >> 11)) >> 16) & 0xbf;
    }
}
struct Plan {
    LinearPath path;
    bool projection_prepared = false;
    int projection_tokens = 0;
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
    size_t l2_bytes = 0, shared_bytes = 0;
    bool hopper = false;
    QuantNorm norm;
    QuantGated gated;
    DeviceArena arena;
    Buffer workspace;
    cublasLtHandle_t handle = nullptr;
    KernelSelector<KernelChoice, KernelOperation> selector;
    PreparedCache<std::tuple<int, int, int>, Plan> plans;
    Impl(DeviceContext &m, Type f)
        : memory(m), stream(m.stream), device_id(m.device), format(f), norm(m, f), gated(m, f),
          arena(m, size_t(8) << 20), workspace(size_t(8) << 20, &arena) {
        cudaDeviceProp property{};
        CUDA(cudaGetDeviceProperties(&property, device_id));
        l2_bytes = property.l2CacheSize;
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
    void run(Plan &p, void *x, const float *xs, void *w, const float *ws, void *y) {
        int m = p.rows, n = p.n, k = p.k;
        if (p.path.kind != LinearKind::Lt) {
            launch_quant_linear(p.path, {m, n, k}, x, xs, w, ws, y, workspace.data(),
                                workspace.size(), stream, format);
            return;
        }
        BLAS(cublasLtMatmulDescSetAttribute(p.operation, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &ws,
                                            sizeof(ws)));
        BLAS(cublasLtMatmulDescSetAttribute(p.operation, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &xs,
                                            sizeof(xs)));
        float alpha = 1, beta = 0;
        BLAS(cublasLtMatmul(handle, p.operation, &alpha, w, p.weight, x, p.input, &beta, y,
                            p.output, y, p.output, &p.algorithm.algo, workspace.data(),
                            workspace.size(), stream));
    }
    void run_projection(Plan &p, const bf16 *x, const unsigned char *w, const float *sw, bf16 *y) {
        launch_quant_projection(p.projection_tokens, {p.rows, p.n, p.k}, x, w, sw, y, stream,
                                format);
    }
    void tune(Plan &p) {
        // Admission only. No timing, allocation or dispatch search in a frame.
        // Rotate a weight bank exceeding L2, so the choice reflects streamed
        // transformer weights rather than repeated reads of a cached matrix.
        auto paths = quant_linear_candidates({p.rows, p.n, p.k},
                                             {hopper, shared_bytes, workspace.size()}, format);
        if (paths.size() == 1)
            return;
        if (!quant_autotune_enabled())
            return;
        size_t matrix = size_t(p.n) * p.k;
        int banks = int(std::clamp<size_t>((2 * l2_bytes + matrix - 1) / matrix, 4, 64));
        WorkspacePlan temporary_size;
        for (size_t bytes : {matrix * banks, size_t(p.rows) * p.k, size_t(p.rows) * 4,
                             size_t(p.n) * 4, size_t(p.rows) * p.n * 2})
            temporary_size.allocate(bytes);
        DeviceArena temporary(memory, temporary_size.size);
        Buffer weights(matrix * banks, &temporary), x(size_t(p.rows) * p.k, &temporary),
            sx(size_t(p.rows) * 4, &temporary), sw(size_t(p.n) * 4, &temporary),
            y(size_t(p.rows) * p.n * 2, &temporary);
        initialize_quant_tuning<<<(weights.size() + 255) / 256, 256, 0, stream>>>(
            weights.ptr<unsigned char>(), weights.size());
        initialize_quant_tuning<<<(x.size() + 255) / 256, 256, 0, stream>>>(x.ptr<unsigned char>(),
                                                                            x.size());
        std::vector<float> ones(std::max(p.rows, p.n));
        for (size_t i = 0; i < ones.size(); ++i)
            ones[i] = .5f + float(i % 11) * .0625f;
        CUDA(cudaMemcpyAsync(sx.data(), ones.data(), sx.size(), cudaMemcpyHostToDevice, stream));
        CUDA(cudaMemcpyAsync(sw.data(), ones.data(), sw.size(), cudaMemcpyHostToDevice, stream));
        CUDA(cudaStreamSynchronize(stream));
        LinearPath best = p.path;
        float best_us = INFINITY;
        std::vector<bf16> expected(size_t(p.rows) * p.n), actual(expected.size());
        bool reference_ready = false;
        for (auto path : paths) {
            p.path = path;
            auto launch = [&](int bank) {
                run(p, x.data(), sx.ptr<float>(), weights.ptr<unsigned char>() + matrix * bank,
                    sw.ptr<float>(), y.data());
            };
            // Initialize library kernels outside graph capture.
            launch(0);
            CUDA(
                cudaMemcpyAsync(actual.data(), y.data(), y.size(), cudaMemcpyDeviceToHost, stream));
            CUDA(cudaStreamSynchronize(stream));
            if (!reference_ready) {
                expected = actual;
                reference_ready = true;
            } else {
                double norm = 0, error = 0;
                for (size_t i = 0; i < actual.size(); ++i) {
                    double v = float(actual[i]), reference = float(expected[i]);
                    if (!std::isfinite(v))
                        throw std::runtime_error("nonfinite quantized tuning output");
                    error += (v - reference) * (v - reference);
                    norm += reference * reference;
                }
                if (error > .0001 * std::max(norm, 1e-30))
                    throw std::runtime_error("quantized tuning candidate failed numerical check");
            }
            float us = tuning::time_linear(stream, launch, banks);
            if (us < best_us) {
                best_us = us;
                best = path;
            }
            TTS_LOG(TTS_LOG_TRACE, "quant.gemm", "{} candidate m={} n={} k={} path={} us={}",
                    quantized_name(format), p.rows, p.n, p.k, path_name(path), us);
        }
        p.path = best;
        TTS_LOG(TTS_LOG_DEBUG, "quant.gemm", "{} selected m={} n={} k={} path={} us={}",
                quantized_name(format), p.rows, p.n, p.k, path_name(best), best_us);
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
            p->path = quant_linear_candidates({rows, n, k},
                                              {hopper, shared_bytes, workspace.size()}, format)
                          .front();
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
        return p;
#else
        throw std::runtime_error("FP8 execution requires CUDA 13");
#endif
    }
    void tune_projection(Plan &p) {
        int rows = p.rows, n = p.n, k = p.k;
        auto candidates = quant_projection_candidates({rows, n, k});
        if (candidates.size() > 1 && quant_autotune_enabled()) {
            size_t matrix = size_t(n) * k;
            int banks = int(std::clamp<size_t>((2 * l2_bytes + matrix - 1) / matrix, 4, 128));
            WorkspacePlan size;
            for (size_t bytes : {matrix * banks, size_t(rows) * k * 2, size_t(rows) * k,
                                 size_t(rows) * 4, size_t(n) * 4, size_t(rows) * n * 2})
                size.allocate(bytes);
            DeviceArena storage(memory, size.size);
            Buffer weights(matrix * banks, &storage), x(size_t(rows) * k * 2, &storage),
                qx(size_t(rows) * k, &storage), sx(size_t(rows) * 4, &storage),
                sw(size_t(n) * 4, &storage), y(size_t(rows) * n * 2, &storage);
            std::vector<bf16> input(size_t(rows) * k);
            std::vector<float> scales(n);
            for (size_t i = 0; i < input.size(); ++i)
                input[i] = bf16(std::sin(i * .037) * (1 + (i / k) % 7));
            for (int i = 0; i < n; ++i)
                scales[i] = .5f + float(i % 11) * .0625f;
            CUDA(cudaMemcpyAsync(x.data(), input.data(), x.size(), cudaMemcpyHostToDevice, stream));
            CUDA(cudaMemcpyAsync(sw.data(), scales.data(), sw.size(), cudaMemcpyHostToDevice,
                                 stream));
            initialize_quant_tuning<<<(weights.size() + 255) / 256, 256, 0, stream>>>(
                weights.ptr<unsigned char>(), weights.size());
            std::vector<bf16> expected(size_t(rows) * n), actual(expected.size());
            int best = 0;
            float best_us = INFINITY;
            for (int tile : candidates) {
                p.projection_tokens = tile;
                auto launch = [&](int bank) {
                    auto *w = weights.ptr<unsigned char>() + matrix * bank;
                    if (tile) {
                        run_projection(p, x.ptr<bf16>(), w, sw.ptr<float>(), y.ptr<bf16>());
                    } else {
                        quantize_rows(format, x.ptr<bf16>(), qx.ptr<unsigned char>(),
                                      sx.ptr<float>(), rows, k, k, stream);
                        run(p, qx.data(), sx.ptr<float>(), w, sw.ptr<float>(), y.data());
                    }
                };
                launch(0);
                CUDA(cudaMemcpyAsync(actual.data(), y.data(), y.size(), cudaMemcpyDeviceToHost,
                                     stream));
                CUDA(cudaStreamSynchronize(stream));
                if (!tile) {
                    expected = actual;
                } else {
                    double norm = 0, error = 0;
                    for (size_t i = 0; i < actual.size(); ++i) {
                        double v = float(actual[i]), ref = float(expected[i]);
                        if (!std::isfinite(v))
                            throw std::runtime_error("nonfinite projection tuning output");
                        error += (v - ref) * (v - ref);
                        norm += ref * ref;
                    }
                    if (error > .0001 * std::max(norm, 1e-30))
                        throw std::runtime_error("projection tuning failed numerical check");
                }
                float us = tuning::time_linear(stream, launch, banks);
                if (us < best_us) {
                    best_us = us;
                    best = tile;
                }
                TTS_LOG(TTS_LOG_TRACE, "quant.projection",
                        "{} candidate m={} n={} k={} tile={} us={}", quantized_name(format), rows,
                        n, k, tile, us);
            }
            p.projection_tokens = best;
            TTS_LOG(TTS_LOG_DEBUG, "quant.projection", "{} selected m={} n={} k={} tile={} us={}",
                    quantized_name(format), rows, n, k, best, best_us);
        }
    }
    ~Impl() {
        if (handle)
            cublasLtDestroy(handle);
    }
};
QuantBackend::QuantBackend(DeviceContext &memory, Type format)
    : impl(std::make_unique<Impl>(memory, format)) {}
QuantBackend::~QuantBackend() = default;
Type QuantBackend::type() const {
    return impl->format;
}
uint64_t QuantBackend::execution_revision() const {
    // Graph nodes own launch parameters, not host descriptors. A new revision
    // additionally prevents replaying an old arithmetic choice after re-tuning.
    return impl->selector.stats().evictions + impl->plans.evictions() +
           impl->norm.execution_revision() + impl->gated.execution_revision();
}
void QuantBackend::begin_admission() {
    impl->plans.begin_admission();
    impl->norm.begin_admission();
    impl->gated.begin_admission();
}
QuantCacheStats QuantBackend::cache_stats() const {
    return {impl->selector.stats(), impl->plans.size()};
}
void QuantBackend::prepare(int rows, int n, int k) {
    impl->plans.prepare(std::make_tuple(rows, n, k), [&] {
        auto p = impl->make_plan(rows, n, k);
        p->path = impl->selector
                      .select(KernelOperation::ScaledLinear, {rows, n, k},
                              [&](KernelProblem problem) {
                                  auto representative = impl->make_plan(
                                      problem.rows, problem.channels, problem.reduction);
                                  impl->tune(*representative);
                                  return KernelChoice{representative->path, 0};
                              })
                      .linear;
        return p;
    });
}
void QuantBackend::prepare_gated(int rows, int width, int kind) {
    impl->gated.prepare(rows, width, kind);
}
void QuantBackend::prepare_norm(int rows, int width, bool gemma) {
    impl->norm.prepare(rows, width, gemma);
}
void QuantBackend::prepare_projection(int rows, int n, int k) {
    prepare(rows, n, k);
    auto &p = *impl->plans.find(std::make_tuple(rows, n, k));
    if (p.projection_prepared)
        return;
    p.projection_tokens =
        impl->selector
            .select(KernelOperation::Bf16InputProjection, {rows, n, k},
                    [&](KernelProblem problem) {
                        auto representative =
                            impl->make_plan(problem.rows, problem.channels, problem.reduction);
                        representative->path = p.path;
                        impl->tune_projection(*representative);
                        return KernelChoice{p.path, representative->projection_tokens};
                    })
            .projection_tokens;
    p.projection_prepared = true;
}
void QuantBackend::project(TensorView x, RowScaledView w, RowScaledView scratch, TensorView y) {
    w.validate();
    scratch.validate();
    if (w.values.type != impl->format || scratch.values.type != impl->format ||
        x.type != Type::BF16 || !x.dense() || x.cols != w.values.cols || x.rows != y.rows ||
        y.type != Type::BF16 || !y.dense() || y.cols != w.values.rows ||
        scratch.values.rows != x.rows || scratch.values.cols != x.cols)
        throw std::invalid_argument("quantized projection shape/type mismatch");
    auto *plan = impl->plans.find(std::make_tuple(y.rows, y.cols, x.cols));
    if (!plan || !plan->projection_prepared)
        throw std::logic_error("quantized projection was not prepared before execution");
    auto &p = *plan;
    if (p.projection_tokens)
        impl->run_projection(p, x.ptr<bf16>(), w.values.ptr<unsigned char>(), w.scales.ptr<float>(),
                             y.ptr<bf16>());
    else {
        quantize(x, scratch);
        linear(scratch, w, y);
    }
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
    auto *plan = impl->plans.find(std::make_tuple(y.rows, y.cols, x.values.cols));
    if (!plan)
        throw std::logic_error("quantized GEMM shape was not prepared before execution");
    impl->run(*plan, x.values.data, x.scales.ptr<float>(), w.values.data, w.scales.ptr<float>(),
              y.data);
}
void QuantBackend::norm(TensorView x, TensorView update, TensorView w, RowScaledView out, float eps,
                        bool gemma, TensorView update_scale) {
    impl->norm.run(x, update, w, out, eps, gemma, update_scale);
}
void QuantBackend::gated(TensorView input, RowScaledView out, int kind) {
    impl->gated.run(input, out, kind);
}
} // namespace tts
