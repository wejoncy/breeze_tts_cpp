#pragma once
#include "io/weights.h"
#include "ops/linear.h"
#include "ops/basic.h"
#include "core/row_capacity.h"
#include "core/forward_context.h"
#include <utility>

namespace tts {
class Linear {
    ops::LinearOperand weight_;
    TensorView bias_;
    Type activation_type_;

    TensorView matrix() const {
        if (const auto *dense = std::get_if<TensorView>(&weight_))
            return *dense;
        return std::get<RowScaledView>(weight_).values;
    }
    Linear(ops::LinearOperand weight, TensorView bias, Type type)
        : weight_(std::move(weight)), bias_(bias), activation_type_(type) {
        auto shape = matrix();
        if (!shape.data || shape.rows < 1 || shape.cols < 1 ||
            (bias.data && (bias.numel() != size_t(shape.rows) || bias.type != type)))
            throw std::invalid_argument("invalid linear weights");
        if (const auto *dense = std::get_if<TensorView>(&weight_)) {
            if ((type != Type::BF16 && type != Type::F16 && type != Type::F32) ||
                dense->type != type)
                throw std::invalid_argument("invalid dense linear dtype/layout");
        } else {
            const auto &quantized = std::get<RowScaledView>(weight_);
            quantized.validate();
            if ((type != Type::BF16 && type != Type::F16) || !quantized.scales.data)
                throw std::invalid_argument("invalid quantized linear activation dtype/scales");
        }
    }

  public:
    struct Workspace {
        Buffer values, scales;
        int max_rows = 0, max_input_channels = 0;
        Workspace() = default;
        Workspace(int rows, int channels, BufferAllocator *allocator)
            : max_rows(rows), max_input_channels(channels) {
            if (rows < 1 || channels < 0)
                throw std::invalid_argument("invalid linear workspace capacity");
            if (!channels)
                return;
            if (size_t(rows) > SIZE_MAX / size_t(channels) ||
                size_t(rows) > SIZE_MAX / sizeof(float))
                throw std::overflow_error("linear workspace size");
            values = Buffer(size_t(rows) * channels, allocator);
            scales = Buffer(size_t(rows) * sizeof(float), allocator);
        }
        RowScaledView view(int rows, int channels, Type format) const {
            if (rows < 1 || rows > max_rows || channels < 1 || channels > max_input_channels ||
                (format != Type::FP8 && format != Type::I8))
                throw std::invalid_argument("linear exceeds initialized quantization workspace");
            // Each projection packs rows at its actual K; max K is capacity, not pitch.
            return {values.view(rows, channels, format), scales.view(rows, 1, Type::F32)};
        }
    };
    Linear(Weights &weights, const std::string &name)
        : Linear(weights.get(name + ".weight"), weights.get(name + ".bias")) {}
    Linear(Weights &weights, const std::string &name, bool quantize)
        : Linear(quantize ? Linear(weights.quantized(name + ".quantized", {name + ".weight"}),
                                    weights.entry(name + ".weight").type, weights.get(name + ".bias"))
                          : Linear(weights, name)) {}
    Linear(TensorView weight, TensorView bias = {})
        : Linear(ops::LinearOperand(weight), bias, weight.type) {}
    Linear(RowScaledView weight, Type activation_type, TensorView bias = {})
        : Linear(ops::LinearOperand(weight), bias, activation_type) {}
    static Linear fused(Weights &weights, const std::string &derived,
                        const std::vector<std::string> &parts, TensorView bias, bool quantize) {
        if (parts.empty())
            throw std::invalid_argument("fused linear has no weight sources");
        return quantize ? Linear(weights.quantized(derived + ".quantized", parts),
                                 weights.entry(parts.front()).type, bias)
                        : Linear(weights.fused(derived, parts), bias);
    }
    Type activation_type() const {
        return activation_type_;
    }
    TensorView forward(TensorView input, ForwardContext &context,
                        const ops::Epilogue &finish = {}) const {
        auto output = context.tensor({context.max_rows, output_channels()}, activation_type_)
                          .slice(0, input.rows);
        return forward(input, output, context, finish);
    }
    TensorView forward(TensorView input, TensorView output, ForwardContext &context,
                        const ops::Epilogue &finish = {}) const {
        if (input.rows < 1 || input.rows > context.max_rows || input.cols != input_channels() ||
            input.type != activation_type_ || output.rows != input.rows ||
            output.cols != output_channels() || output.type != activation_type_)
            throw std::invalid_argument("linear input shape/type");
        Workspace workspace(context.max_rows, is_quantized() ? input_channels() : 0,
                            &context.storage);
        if (context.mode == ForwardContext::Prepare && input.data && output.data)
            prepare_range(input, output, context.max_rows, finish.activation);
        else if (context.mode == ForwardContext::Execute)
            forward(workspace, input, output, finish);
        return output;
    }
    TensorView forward_segments(TensorView input, TensorView segments, TensorView output,
                                 ForwardContext &context) const {
        Workspace workspace(context.max_rows, is_quantized() ? input_channels() : 0,
                            &context.storage);
        if (context.mode == ForwardContext::Plan) return output;
        if (context.mode == ForwardContext::Prepare) {
            if (input.data && output.data)
                prepare_range(input.slice(0), output.slice(0), context.max_rows);
            return output;
        }
        if (segments.device != -1 || segments.type != Type::I32 || segments.cols != 3)
            throw std::invalid_argument("projection segments must contain host I32 row ranges");
        if (!segments.rows) {
            forward(workspace, input, output);
            return output;
        }
        int contiguous_rows = 0;
        for (int segment_index = 0; segment_index < segments.rows; ++segment_index) {
            const auto *range = segments.ptr<int>() + size_t(segment_index) * segments.ld();
            if (range[0] != contiguous_rows || range[2] != contiguous_rows) break;
            contiguous_rows += range[1];
        }
        if (contiguous_rows == input.rows && output.rows == input.rows) {
            forward(workspace, input, output);
            return output;
        }
        for (int segment_index = 0; segment_index < segments.rows; ++segment_index) {
            const auto *range = segments.ptr<int>() + size_t(segment_index) * segments.ld();
            forward(workspace, input.slice(range[0], range[1]), output.slice(range[2], range[1]));
        }
        return output;
    }
    bool is_quantized() const {
        return std::holds_alternative<RowScaledView>(weight_);
    }
    TensorView dense_weight() const {
        if (const auto *dense = std::get_if<TensorView>(&weight_))
            return *dense;
        throw std::logic_error("quantized linear has no dense weight");
    }
    TensorView bias() const {
        return bias_;
    }
    int input_channels() const {
        return matrix().cols;
    }
    int output_channels() const {
        return matrix().rows;
    }
    void prepare_range(int max_rows,
                       ops::Activation activation = ops::Activation::Identity) const {
        if (!is_quantized())
            return;
        if (max_rows < 1)
            throw std::invalid_argument("invalid linear preparation capacity");
        for (int rows = 1;;) {
            int capacity = row_capacity(rows);
            ops::Epilogue finish;
            finish.bias = bias_;
            finish.activation = activation;
            ops::prepare_quantized(matrix(), capacity, output_channels(), input_channels(),
                                     activation_type_, finish);
            if (capacity >= max_rows)
                break;
            rows = capacity + 1;
        }
    }
    void prepare_range(TensorView input, TensorView output, int max_rows,
                       ops::Activation activation = ops::Activation::Identity) const {
        validate_types(input, output);
        if (max_rows < 1 || input.rows < 1 || input.rows > max_rows || input.rows != output.rows ||
            input.cols != input_channels() || output.cols != output_channels())
            throw std::invalid_argument("invalid linear planning views/capacity");
        if (is_quantized()) {
            prepare_range(max_rows, activation);
            return;
        }
        if (activation == ops::Activation::GeluTanh &&
            ops::prepare_matmul_gelu(input, dense_weight(), bias_, output, max_rows))
            return;
        ops::prepare_matmul(input, dense_weight(), output, false, 0, 1, 1, max_rows);
    }
    void project(TensorView input, TensorView output) const {
        apply(input, output, {}, {}, nullptr);
    }
    void project(Workspace &workspace, TensorView input, TensorView output) const {
        apply(input, output, {}, {}, &workspace);
    }
    RowScaledView quantized_view(const Workspace &workspace, int rows) const {
        if (const auto *quantized = std::get_if<RowScaledView>(&weight_))
            return workspace.view(rows, input_channels(), quantized->values.type);
        throw std::logic_error("dense linear does not consume quantized activations");
    }
    void project(RowScaledView input, TensorView output) const {
        if (!is_quantized() || output.type != activation_type_)
            throw std::invalid_argument("quantized linear activation dtype mismatch");
        ops::linear(input, weight_, {}, output);
    }
    RowScaledView normalize_input(Workspace &workspace, TensorView input,
                                   TensorView norm_weight, TensorView norm_bias, float epsilon,
                                   const ops::NormFusion &fusion, TensorView retained = {}) const {
        if (input.type != activation_type_ || input.cols != input_channels())
            throw std::invalid_argument("linear normalization input mismatch");
        auto output = quantized_view(workspace, input.rows);
        ops::layer_norm_quantize( input, output, norm_weight, norm_bias, epsilon, fusion,
                                  retained);
        return output;
    }
    bool normalized_forward(TensorView input, TensorView normalized, TensorView output,
                            TensorView norm_weight, TensorView norm_bias, float epsilon,
                            const ops::NormFusion &fusion, ops::Epilogue epilogue = {},
                            const QkvProjectionOutput *writeback = nullptr) const {
        validate_types(input, output);
        return ops::normalized_linear(input, weight_, bias_, normalized, output, norm_weight,
                                      norm_bias, epsilon, fusion, epilogue, writeback);
    }
    bool normalized_forward(Workspace &workspace, TensorView input,
                             TensorView normalized, TensorView output, TensorView norm_weight,
                             TensorView norm_bias, float epsilon, const ops::NormFusion &fusion,
                             ops::Epilogue epilogue = {}, const QkvProjectionOutput *writeback = nullptr,
                             RowScaledView quantized_output = {}) const {
        validate_types(input, output);
        if (is_quantized()) {
            if (writeback)
                return false;
            if (ops::normalized_linear(input, weight_, bias_, normalized, output, norm_weight,
                                       norm_bias, epsilon, fusion, epilogue, nullptr, quantized_output))
                return true;
            auto normalized_input = normalize_input(workspace, input, norm_weight, norm_bias, epsilon,
                                                      fusion, normalized);
            ops::linear(normalized_input, weight_, bias_, output, epilogue, {}, quantized_output);
            return true;
        }
        return !quantized_output.values.data &&
               normalized_forward(input, normalized, output, norm_weight, norm_bias, epsilon, fusion,
                                    epilogue, writeback);
    }
    void forward(TensorView input, TensorView output, ops::Epilogue epilogue = {}) const {
        apply(input, output, bias_, epilogue, nullptr);
    }
    void forward(Workspace &workspace, TensorView input, TensorView output,
                 ops::Epilogue epilogue = {}, RowScaledView quantized_output = {}) const {
        apply(input, output, bias_, epilogue, &workspace, quantized_output);
    }

  private:
    void validate_types(TensorView input, TensorView output) const {
        if (input.type != activation_type_ ||
            (output.type != activation_type_ && (is_quantized() || output.type != Type::F32)))
            throw std::invalid_argument("linear activation dtype mismatch");
    }
    void apply(TensorView input, TensorView output, TensorView bias, ops::Epilogue epilogue,
               Workspace *workspace, RowScaledView quantized_output = {}) const {
        const auto *quantized = std::get_if<RowScaledView>(&weight_);
        validate_types(input, output);
        RowScaledView scratch;
        if (quantized) {
            if (!workspace)
                throw std::logic_error("quantized linear requires an explicit workspace");
            scratch = workspace->view(input.rows, input.cols, quantized->values.type);
        }
        ops::linear(input, weight_, bias, output, epilogue, scratch, quantized_output);
    }
};
} // namespace tts
