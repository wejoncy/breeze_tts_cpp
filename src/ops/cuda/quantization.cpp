#include "ops/cuda/quantization.h"
#include "ops/cuda/lt_plan.h"
#include "ops/cuda/cuda_check.h"
#include "ops/kernels/common/chip_execution.h"
#include "ops/cuda/quantization_policy.h"
#include "core/row_capacity.h"
#include <map>
#include <limits>
#include "ops/kernels/quant/quant_dispatch.h"
#include "ops/kernels/sequence/norm_projection.h"
#include "runtime/prepared_cache.h"
#include "support/log.h"
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cublasLt.h>
#include <tuple>

namespace tts {
namespace {
// CPU-only policy. The injected policy selects a configuration for each row class.
// No tensor, CUDA, allocator, model, graph, or serving dependency is allowed here.
// Choices live for the selector lifetime. Row classes grow logarithmically;
// Choices remain available to every prepared plan in the class.
// GPU plans/descriptors have their own bounded cache and do not own these choices.
template <class Choice> class KernelSelector {
    using Key = std::tuple<int, int, int, int, int>;
    std::map<Key, Choice> entries_;
    unsigned bins_per_octave_;
    SelectionStats stats_;

  public:
    explicit KernelSelector(unsigned bins_per_octave = 8) : bins_per_octave_(bins_per_octave) {
        if (!bins_per_octave)
            throw std::invalid_argument("selection class resolution must be positive");
    }
    static KernelProblem representative(KernelProblem p, unsigned bins_per_octave = 8) {
        if (p.rows < 1 || p.channels < 1 || p.reduction < 1)
            throw std::invalid_argument("invalid kernel problem");
        p.rows = row_capacity(p.rows, bins_per_octave);
        return p;
    }
    template <class Search>
    Choice select(KernelProblem actual, Search &&search) {
        auto p = representative(actual, bins_per_octave_);
        Key key{p.rows, p.channels, p.reduction, p.input_group, p.weight_group};
        auto it = entries_.find(key);
        if (it != entries_.end()) {
            ++stats_.hits;
            return it->second;
        }
        // Exceptions do not publish partial/invalid selections.
        Choice choice = search(p);
        entries_.emplace(key, choice);
        ++stats_.searches;
        return choice;
    }
    SelectionStats stats() const {
        auto result = stats_;
        result.entries = entries_.size();
        return result;
    }
};

struct Plan : LtPlan {
    LinearPath path, native_path;
    bool lt_compatible = true, finish_available = false;
    int rows = 0, channels = 0, reduction = 0;
    Type output_type = Type::BF16;
    QuantizationGroups groups;
};
using PlanKey = std::tuple<int, int, int, Type, int, int>;
using FinishKey = std::tuple<PlanKey, ops::Activation, uintptr_t>;
bool simple_finish(const ops::Epilogue &epilogue) {
    return (epilogue.activation == ops::Activation::Relu ||
            epilogue.activation == ops::Activation::GeluTanh) &&
           !epilogue.residual.data && !epilogue.gate.data && !epilogue.condition.data &&
           !epilogue.raw.data && !epilogue.residual_rows && !epilogue.valid_rows_per_sequence &&
           !epilogue.packed && !epilogue.condition_sources && !epilogue.output_rows &&
           epilogue.input_repeat == 1 && epilogue.value_scale == 1 && epilogue.output_scale == 1;
}
void check_finish_bias(const ops::Epilogue &epilogue, int channels, Type type) {
    if (epilogue.bias.data && (epilogue.bias.type != type || !epilogue.bias.dense() ||
                               epilogue.bias.numel() != size_t(channels)))
        throw std::invalid_argument("quantized linear finish bias mismatch");
}
FinishKey finish_key(int rows, int channels, int reduction, Type type,
                     const ops::Epilogue &epilogue) {
    auto address = reinterpret_cast<uintptr_t>(epilogue.bias.data);
    auto alignment = std::min<uintptr_t>(256, address & (~address + 1));
    return {{row_capacity(rows), channels, reduction, type, 0, 0}, epilogue.activation, alignment};
}
void check_linear(Type format, RowScaledView input, RowScaledView weight, TensorView output) {
    input.validate();
    weight.validate();
    if (input.values.type != format || weight.values.type != format ||
        input.rotation_size != weight.rotation_size || input.values.cols != weight.values.cols ||
        input.values.rows != output.rows || weight.values.rows != output.cols ||
        (output.type != Type::BF16 && output.type != Type::F16) || !output.dense())
        throw std::invalid_argument("quantized linear dimensions/type mismatch");
}
LinearFinishWork finish_work(const ops::Epilogue &finish) {
    return {(finish.activation == ops::Activation::GeluTanh ? 9. : 1.) + 1 +
            (finish.bias.data ? 4 : 0)};
}
} // namespace
struct QuantizedGemm::Impl {
    cudaStream_t stream;
    Type format;
    size_t shared_bytes = 0;
    ChipModel chip;
    bool hopper = false;
    int normalized_capacity = 0;
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
    Impl(cudaStream_t execution_stream, cublasLtHandle_t lt, Buffer &scratch, Type storage_format)
        : stream(execution_stream), format(storage_format), workspace(scratch), handle(lt) {
        const int device_id = scratch.device();
        cudaDeviceProp property{};
        CUDA(cudaGetDeviceProperties(&property, device_id));
        chip.sms = property.multiProcessorCount;
        normalized_capacity = ops::normalized_quantized_linear_capacity(storage_format, chip.sms);
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
        chip.scalar_ops_per_us =
            (storage_format == Type::I8 ? 512. : scalar_lanes * 2.) * sm_clock / 1000;
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
    void run(Plan &plan, void *input, const float *input_scales, void *weight,
             const float *weight_scales, void *output, int actual_rows = 0,
             const ops::Epilogue *finish = nullptr) {
        int rows = actual_rows ? actual_rows : plan.rows, channels = plan.channels,
            reduction = plan.reduction;
        bool unaligned_groups = (plan.groups.activation || plan.groups.weight) && rows % 4;
        if (plan.path.kind != LinearKind::Lt || unaligned_groups) {
            launch_quant_linear(
                unaligned_groups && plan.path.kind == LinearKind::Lt ? plan.native_path : plan.path,
                {rows, channels, reduction, plan.groups.activation, plan.groups.weight}, input,
                input_scales, weight, weight_scales, output, workspace.data(), workspace.size(),
                stream, format, plan.output_type, finish);
            return;
        }
        uint64_t columns = rows;
        BLAS(cublasLtMatrixLayoutSetAttribute(plan.input, CUBLASLT_MATRIX_LAYOUT_COLS, &columns,
                                              sizeof(columns)));
        BLAS(cublasLtMatrixLayoutSetAttribute(plan.output, CUBLASLT_MATRIX_LAYOUT_COLS, &columns,
                                              sizeof(columns)));
        BLAS(cublasLtMatmulDescSetAttribute(plan.operation, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                            &weight_scales, sizeof(weight_scales)));
        BLAS(cublasLtMatmulDescSetAttribute(plan.operation, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                            &input_scales, sizeof(input_scales)));
        if (finish && finish->bias.data)
            BLAS(cublasLtMatmulDescSetAttribute(plan.operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                &finish->bias.data, sizeof(finish->bias.data)));
        float alpha = 1, beta = 0;
        BLAS(cublasLtMatmul(handle, plan.operation, &alpha, weight, plan.weight, input, plan.input,
                            &beta, output, plan.output, output, plan.output, &plan.algorithm.algo,
                            workspace.data(), workspace.size(), stream));
    }
    std::unique_ptr<Plan> make_plan(int rows, int channels, int reduction, Type output_type,
                                    const ops::Epilogue *finish = nullptr,
                                    QuantizationGroups groups = {}) {
        output_index(output_type);
        if (rows < 1 || channels < 1 || reduction < 1 || reduction % 16 || channels % 16)
            throw std::invalid_argument("quantized GEMM unsupported shape");
        auto plan = std::make_unique<Plan>();
        plan->rows = rows;
        plan->channels = channels;
        plan->reduction = reduction;
        plan->output_type = output_type;
        plan->groups = groups;
        if (format == Type::I8) {
            // Signed byte products accumulate exactly in INT32, including split-K.
            if (reduction > INT32_MAX / (128 * 128))
                throw std::invalid_argument("INT8 reduction can overflow INT32");
            return plan;
        }
#if CUBLAS_VERSION >= 130000
        BLAS(cublasLtMatmulDescCreate(&plan->operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        int8_t fast_accum = 0;
        BLAS(cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_FAST_ACCUM,
                                            &fast_accum, sizeof(fast_accum)));
        auto transpose = CUBLAS_OP_T;
        BLAS(cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_TRANSA,
                                            &transpose, sizeof(transpose)));
        auto mode = groups.activation || groups.weight ? CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F
                                                       : CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
        if ((groups.activation || groups.weight) &&
            (groups.activation != 128 || groups.weight != 128 || rows < 4)) {
            plan->lt_compatible = false;
            return plan;
        }
        BLAS(cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                            &mode, sizeof(mode)));
        BLAS(cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                            &mode, sizeof(mode)));
        // Outer-vector heuristics require non-null device scale pointers. Planning
        // does not execute the GEMM; bind aligned storage, then replace both pointers
        // with the actual row scales in linear() before every launch/capture.
        const void *planning_scale = workspace.data();
        BLAS(cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                            &planning_scale, sizeof(planning_scale)));
        BLAS(cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                            &planning_scale, sizeof(planning_scale)));
        if (finish) {
            auto epilogue =
                finish->activation == ops::Activation::GeluTanh
                    ? (finish->bias.data ? CUBLASLT_EPILOGUE_GELU_BIAS : CUBLASLT_EPILOGUE_GELU)
                    : (finish->bias.data ? CUBLASLT_EPILOGUE_RELU_BIAS : CUBLASLT_EPILOGUE_RELU);
            BLAS(cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                &epilogue, sizeof(epilogue)));
            if (finish->bias.data) {
                auto bias_type = output_type == Type::F16 ? CUDA_R_16F : CUDA_R_16BF;
                BLAS(cublasLtMatmulDescSetAttribute(plan->operation,
                                                    CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                    &finish->bias.data, sizeof(finish->bias.data)));
                BLAS(cublasLtMatmulDescSetAttribute(plan->operation,
                                                    CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type,
                                                    sizeof(bias_type)));
            }
        }
        int library_rows = groups.activation ? (rows + 3) / 4 * 4 : rows;
        BLAS(cublasLtMatrixLayoutCreate(&plan->weight, CUDA_R_8F_E4M3, reduction, channels,
                                        reduction));
        BLAS(cublasLtMatrixLayoutCreate(&plan->input, CUDA_R_8F_E4M3, reduction, library_rows,
                                        reduction));
        BLAS(cublasLtMatrixLayoutCreate(&plan->output,
                                        output_type == Type::F16 ? CUDA_R_16F : CUDA_R_16BF,
                                        channels, library_rows, channels));
        if (!plan->select(handle, workspace.size())) {
            plan->lt_compatible = false;
            return plan;
        }
        // A selected algorithm must support every actual M in this capacity
        // class. This is a host-only initialization check, never a serving query.
        int lower = rows;
        while (lower > 1 && row_capacity(lower - 1) == rows)
            --lower;
        for (int actual = lower; actual <= rows; ++actual) {
            if (groups.activation && actual % 4) {
                continue;
            }
            uint64_t columns = actual;
            BLAS(cublasLtMatrixLayoutSetAttribute(plan->input, CUBLASLT_MATRIX_LAYOUT_COLS,
                                                  &columns, sizeof(columns)));
            BLAS(cublasLtMatrixLayoutSetAttribute(plan->output, CUBLASLT_MATRIX_LAYOUT_COLS,
                                                  &columns, sizeof(columns)));
            cublasLtMatmulHeuristicResult_t checked{};
            auto status = cublasLtMatmulAlgoCheck(handle, plan->operation, plan->weight,
                                                  plan->input, plan->output, plan->output,
                                                  &plan->algorithm.algo, &checked);
            if (status != CUBLAS_STATUS_SUCCESS || checked.state != CUBLAS_STATUS_SUCCESS ||
                checked.workspaceSize > workspace.size())
                plan->lt_compatible = false;
        }
        return plan;
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
        Buffer scratch(4 * elements + 2 * extent * sizeof(float));
        auto input = scratch.view(extent, extent, format);
        auto weight = scratch.view(extent, extent, format, elements);
        auto input_scales = scratch.view(extent, 1, Type::F32, 4 * elements);
        auto weight_scales =
            scratch.view(extent, 1, Type::F32, 4 * elements + extent * sizeof(float));
        CUDA(cudaMemsetAsync(scratch.data(), 0, scratch.size(), stream));
        for (auto type : {Type::BF16, Type::F16}) {
            auto output = scratch.view(extent, extent, type, 2 * elements);
            for (auto activation :
                 {ops::Activation::Identity, ops::Activation::Relu, ops::Activation::GeluTanh})
                for (bool bias : {false, true}) {
                    if (activation == ops::Activation::Identity && bias)
                        continue;
                    ops::Epilogue finish;
                    finish.activation = activation;
                    // Read-only zero backing; bias never aliases the output.
                    if (bias)
                        finish.bias = scratch.view(1, extent, type);
                    const auto *epilogue =
                        activation == ops::Activation::Identity ? nullptr : &finish;
                    auto plan = make_plan(extent, extent, extent, type, epilogue);
                    if (plan->lt_compatible)
                        run(*plan, input.data, input_scales.ptr<float>(), weight.data,
                            weight_scales.ptr<float>(), output.data, 0, epilogue);
                }
        }
        CUDA(cudaStreamSynchronize(stream));
    }
};
QuantizedGemm::QuantizedGemm(cudaStream_t stream, cublasLtHandle_t handle, Buffer &workspace,
                             Type format)
    : impl(std::make_unique<Impl>(stream, handle, workspace, format)) {
    impl->initialize_library();
}
QuantizedGemm::~QuantizedGemm() = default;
void QuantizedGemm::seal() {
    impl->plans.seal();
    impl->finishes.seal();
    impl->finish_resources.seal();
}
QuantCacheStats QuantizedGemm::cache_stats() const {
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
void QuantizedGemm::prepare(int rows, int channels, int reduction, Type output_type,
                            QuantizationGroups groups) {
    auto &selector = impl->selectors[Impl::output_index(output_type)];
    rows = row_capacity(rows);
    impl->plans.prepare(
        std::make_tuple(rows, channels, reduction, output_type, groups.activation, groups.weight),
        [&] {
            auto plan = impl->make_plan(rows, channels, reduction, output_type, nullptr, groups);
            plan->path = selector.select(
                {rows, channels, reduction, groups.activation, groups.weight},
                [&](KernelProblem problem) {
                    auto path = select_linear(
                        problem, {impl->hopper, impl->shared_bytes, impl->workspace.size()},
                        impl->format, impl->chip, plan->lt_compatible, plan->work);
                    TTS_LOG(TTS_LOG_DEBUG, "quant.gemm",
                            "{} planned rows={} channels={} reduction={} output={} path={}",
                            quantized_name(impl->format), problem.rows, channels, reduction,
                            output_type == Type::F16 ? "f16" : "bf16", path_name(path));
                    return path;
                });
            if (plan->path.kind == LinearKind::Lt && (groups.activation || groups.weight))
                plan->native_path =
                    select_linear({rows, channels, reduction, groups.activation, groups.weight},
                                  {impl->hopper, impl->shared_bytes, impl->workspace.size()},
                                  impl->format, impl->chip, false);
            return plan;
        });
}
void QuantizedGemm::prepare_finish(int rows, int channels, int reduction, Type output_type,
                                   const ops::Epilogue &finish, QuantizationGroups groups) {
    if (groups.activation || groups.weight)
        return;
    if (!simple_finish(finish) || (impl->format == Type::I8 && !impl->hopper))
        return;
    check_finish_bias(finish, channels, output_type);
    // The native column broadcast uses up to one 128-bit vector load.
    if (impl->format == Type::I8 && reinterpret_cast<uintptr_t>(finish.bias.data) % 16)
        return;
    rows = row_capacity(rows);
    auto *ordinary = impl->plans.find({rows, channels, reduction, output_type, 0, 0});
    if (!ordinary)
        throw std::logic_error("quantized finish requires a prepared projection");
    if (impl->format == Type::FP8 && ordinary->path.kind != LinearKind::Lt)
        return;
    impl->finishes.prepare(finish_key(rows, channels, reduction, output_type, finish), [&] {
        auto plan = impl->make_plan(rows, channels, reduction, output_type,
                                    impl->format == Type::FP8 ? &finish : nullptr);
        KernelProblem problem{rows, channels, reduction};
        auto work = finish_work(finish);
        double best = linear_pipeline_cost(problem, ordinary->path, impl->format, impl->chip, work,
                                           false, ordinary->work);
        int best_registers = std::numeric_limits<int>::max();
        auto consider = [&](LinearPath path, LibraryWork library, KernelResources resources) {
            double cost = linear_pipeline_cost(problem, path, impl->format, impl->chip, work, true,
                                               library, resources);
            if (cost < best ||
                (plan->finish_available && cost == best && resources.registers < best_registers)) {
                plan->path = path;
                plan->finish_available = true;
                best = cost;
                best_registers = resources.registers;
            }
        };
        if (impl->format == Type::FP8) {
            if (plan->lt_compatible)
                consider({}, plan->work, {});
        } else {
            auto paths = quant_linear_candidates(
                problem, {impl->hopper, impl->shared_bytes, impl->workspace.size()}, impl->format);
            for (auto path : paths) {
                if (path.kind != LinearKind::Hopper || path.splits != 1)
                    continue;
                auto &resources = impl->finish_resources.prepare(
                    {path.kind, path.channels, path.tokens, output_type}, [&] {
                        return std::make_unique<KernelResources>(
                            quant_linear_finish_resources(path, impl->format, output_type));
                    });
                consider(path, {}, resources);
            }
        }
        if (plan->finish_available)
            TTS_LOG(TTS_LOG_DEBUG, "quant.finish",
                    "{} planned rows={} channels={} reduction={} output={} path={} "
                    "estimated_pipeline_us={}",
                    quantized_name(impl->format), rows, channels, reduction,
                    output_type == Type::F16 ? "f16" : "bf16", path_name(plan->path), best);
        return plan;
    });
}
void QuantizedGemm::linear(RowScaledView input, RowScaledView weight, TensorView output) {
    check_linear(impl->format, input, weight, output);
    auto *plan =
        impl->plans.find(std::make_tuple(row_capacity(output.rows), output.cols, input.values.cols,
                                         output.type, input.group_size, weight.group_size));
    if (!plan)
        throw std::logic_error("quantized GEMM shape was not prepared before execution");
    impl->run(*plan, input.values.data, input.scales.ptr<float>(), weight.values.data,
              weight.scales.ptr<float>(), output.data, output.rows);
}
bool QuantizedGemm::linear_finish(RowScaledView input, RowScaledView weight, TensorView output,
                                  const ops::Epilogue &finish) {
    if (!simple_finish(finish) || input.group_size || weight.group_size)
        return false;
    check_linear(impl->format, input, weight, output);
    check_finish_bias(finish, output.cols, output.type);
    auto *plan = impl->finishes.find(
        finish_key(output.rows, output.cols, input.values.cols, output.type, finish));
    if (!plan || !plan->finish_available)
        return false;
    impl->run(*plan, input.values.data, input.scales.ptr<float>(), weight.values.data,
              weight.scales.ptr<float>(), output.data, output.rows, &finish);
    return true;
}
bool QuantizedGemm::normalized_linear(TensorView input, TensorView normalized,
                                      TensorView norm_weight, TensorView norm_bias, float epsilon,
                                      const ops::NormFusion &fusion, RowScaledView weight,
                                      TensorView output, const ops::Epilogue &epilogue,
                                      RowScaledView encoded, const QkvProjectionOutput *writeback) {
    if (epilogue.packed || epilogue.condition_sources || epilogue.output_rows ||
        weight.group_size || encoded.group_size || weight.rotation_size || encoded.rotation_size)
        return false;
    auto *plan = impl->plans.find(std::make_tuple(row_capacity(input.rows), weight.values.rows,
                                                  input.cols, output.type, 0, 0));
    if (!plan || plan->path.kind != LinearKind::Gemv || weight.values.type != impl->format)
        return false;
    return ops::normalized_quantized_linear(impl->stream, input, normalized, norm_weight, norm_bias,
                                            epsilon, fusion, weight, output, epilogue,
                                            impl->normalized_capacity, encoded, writeback);
}
} // namespace tts
