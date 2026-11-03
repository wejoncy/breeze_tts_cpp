#include "ops/linear.h"
#include "ops/backend.h"
#include "ops/kernels/sequence/norm_projection.h"
#include <type_traits>

namespace tts::ops {
namespace {
void check_linear_output(TensorView output, TensorView bias, const Epilogue &e) {
    if (e.bias.data)
        throw std::invalid_argument("linear already owns its bias");
    for (auto source : {bias, e.residual, e.gate, e.condition})
        if (source.overlaps(output))
            throw std::invalid_argument("linear epilogue source aliases its projection output");
}
} // namespace
void linear(const LinearOperand &input, const LinearOperand &weight, TensorView bias,
            TensorView output, Epilogue e, RowScaledView scratch, RowScaledView quantized_output) {
    auto &gpu = backend_for(output, {bias, e.residual, e.gate, e.condition, e.raw,
                                    scratch.values, scratch.scales,
                                    quantized_output.values, quantized_output.scales});
    auto device = [&](const LinearOperand &operand) {
        std::visit([&](auto matrix) {
            if constexpr (std::is_same_v<decltype(matrix), TensorView>)
                backend_for(output, {matrix});
            else
                backend_for(output, {matrix.values, matrix.scales});
        }, operand);
    };
    device(input);
    device(weight);
    check_linear_output(output, bias, e);
    e.bias = bias;
    if (quantized_output.values.data)
        validate_pointwise_quantize(output, quantized_output, e);
    bool fused = std::visit([&](auto matrix) {
        if constexpr (std::is_same_v<decltype(matrix), TensorView>) {
            const auto *dense = std::get_if<TensorView>(&input);
            if (!dense)
                throw std::invalid_argument("dense linear requires a dense activation");
            if (!quantized_output.values.data && !e.packed && !e.condition_sources && !e.output_rows &&
                e.activation == Activation::GeluTanh &&
                !e.residual.data && !e.gate.data &&
                !e.condition.data && !e.raw.data && e.value_scale == 1 && e.output_scale == 1 &&
                !e.residual_rows && !e.valid_rows_per_sequence && e.input_repeat == 1 &&
                e.rows_per_sequence > 0 && output.rows % e.rows_per_sequence == 0 &&
                gpu.linear_gelu(*dense, matrix, bias, output))
                return true;
            gpu.linear(*dense, matrix, output);
        } else {
            matrix.validate();
            RowScaledView activation;
            if (const auto *dense = std::get_if<TensorView>(&input)) {
                gpu.quantized().quantize(*dense, scratch);
                activation = scratch;
            } else {
                activation = std::get<RowScaledView>(input);
            }
            if (activation.values.cols != matrix.values.cols ||
                output.rows != activation.values.rows || output.cols != matrix.values.rows ||
                !output.dense())
                throw std::invalid_argument("quantized linear dimensions/layout mismatch");
            if (!e.packed && !e.condition_sources && !e.output_rows && quantized_output.values.data &&
                quantized_output.values.type == matrix.values.type &&
                gpu.quantized().linear_finish(activation, matrix, output, e)) {
                gpu.quantized().quantize(output, quantized_output);
                return true;
            }
            gpu.quantized().linear(activation, matrix, output);
        }
        return false;
    }, weight);
    if (fused)
        return;
    if (quantized_output.values.data) {
        pointwise_quantize(gpu.stream, output, quantized_output, e, {}, gpu.multiprocessors());
    } else if (bias.data || e.residual.data || e.gate.data || e.condition.data || e.raw.data ||
        e.activation != Activation::Identity || e.value_scale != 1 || e.output_scale != 1 ||
        e.valid_rows_per_sequence || e.packed || e.condition_sources || e.output_rows) {
        pointwise(gpu.stream, output, output, e);
    }
}
bool normalized_linear(TensorView input, const LinearOperand &weight, TensorView bias,
                       TensorView normalized, TensorView output, TensorView norm_weight,
                       TensorView norm_bias, float epsilon, const NormFusion &fusion,
                       Epilogue e, const QkvProjectionOutput *writeback,
                       RowScaledView quantized_output) {
    auto &gpu = backend_for(input, {bias, normalized, output, norm_weight, norm_bias,
                                   fusion.residual, fusion.projection_bias,
                                   quantized_output.values, quantized_output.scales});
    check_linear_output(output, bias, e);
    if (writeback && writeback->bias.data != bias.data)
        throw std::invalid_argument("linear already owns its bias");
    if (e.packed || e.condition_sources || e.output_rows)
        return false;
    if (const auto *quantized = std::get_if<RowScaledView>(&weight)) {
        if (writeback && quantized_output.values.data) return false;
        auto project = [&](const Epilogue &epilogue) {
            return gpu.quantized().normalized_linear(input, normalized, norm_weight, norm_bias,
                epsilon, fusion, *quantized, output, epilogue, {}, writeback);
        };
        if (quantized_output.values.data) {
            e.bias = bias;
            validate_pointwise_quantize(output, quantized_output, e);
            if (quantized_output.values.type == quantized->values.type && project(e)) {
                gpu.quantized().quantize(output, quantized_output);
            } else {
                if (!project({})) return false;
                pointwise_quantize(gpu.stream, output, quantized_output, e, {},
                                   gpu.multiprocessors());
            }
            return true;
        }
        if (!writeback) e.bias = bias;
        return project(e);
    }
    if (quantized_output.values.data) return false;
    if (!writeback) e.bias = bias;
    return normalized_linear(gpu.stream, input, normalized, norm_weight, norm_bias, epsilon,
                             fusion, std::get<TensorView>(weight), output, e,
                             gpu.normalized_linear_capacity(writeback != nullptr), writeback);
}
} // namespace tts::ops
