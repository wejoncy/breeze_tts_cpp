#pragma once
#include "layers/weights.h"
#include "ops/linear.h"
#include "core/row_capacity.h"
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
        auto &gpu = backend_for(matrix());
        if (!is_quantized())
            return;
        if (max_rows < 1)
            throw std::invalid_argument("invalid linear preparation capacity");
        for (int rows = 1;;) {
            int capacity = row_capacity(rows);
            gpu.quantized().prepare(capacity, output_channels(), input_channels(), activation_type_);
            ops::Epilogue finish;
            finish.bias = bias_;
            finish.activation = activation;
            gpu.quantized().prepare_finish(capacity, output_channels(), input_channels(),
                                           activation_type_, finish);
            if (capacity >= max_rows)
                break;
            rows = capacity + 1;
        }
    }
    void prepare_range(TensorView x, TensorView y, int max_rows,
                       ops::Activation activation = ops::Activation::Identity) const {
        auto &gpu = backend_for(x, {matrix(), y});
        validate_types(x, y);
        if (max_rows < 1 || x.rows < 1 || x.rows > max_rows || x.rows != y.rows ||
            x.cols != input_channels() || y.cols != output_channels())
            throw std::invalid_argument("invalid linear planning views/capacity");
        if (is_quantized()) {
            prepare_range(max_rows, activation);
            return;
        }
        if (activation == ops::Activation::GeluTanh &&
            gpu.prepare_linear_gelu(x, dense_weight(), bias_, y, max_rows))
            return;
        gpu.prepare_linear(x, dense_weight(), y, false, 0, 1, 1, max_rows);
    }
    void project(TensorView x, TensorView y) const {
        apply(x, y, {}, {}, nullptr);
    }
    void project(Workspace &workspace, TensorView x, TensorView y) const {
        apply(x, y, {}, {}, &workspace);
    }
    RowScaledView quantized_view(const Workspace &workspace, int rows) const {
        if (const auto *quantized = std::get_if<RowScaledView>(&weight_))
            return workspace.view(rows, input_channels(), quantized->values.type);
        throw std::logic_error("dense linear does not consume quantized activations");
    }
    void project(RowScaledView x, TensorView y) const {
        if (!is_quantized() || y.type != activation_type_)
            throw std::invalid_argument("quantized linear activation dtype mismatch");
        ops::linear(x, weight_, {}, y);
    }
    RowScaledView normalize_input(Workspace &workspace, TensorView x,
                                   TensorView norm_weight, TensorView norm_bias, float epsilon,
                                   const ops::NormFusion &fusion, TensorView retained = {}) const {
        if (x.type != activation_type_ || x.cols != input_channels())
            throw std::invalid_argument("linear normalization input mismatch");
        auto output = quantized_view(workspace, x.rows);
        ops::layer_norm_quantize(backend_for(x).stream, x, output, norm_weight, norm_bias, epsilon, fusion,
                                  retained);
        return output;
    }
    bool normalized_forward(TensorView x, TensorView normalized, TensorView y,
                            TensorView norm_weight, TensorView norm_bias, float epsilon,
                            const ops::NormFusion &fusion, ops::Epilogue e = {},
                            const QkvProjectionOutput *writeback = nullptr) const {
        validate_types(x, y);
        return ops::normalized_linear(x, weight_, bias_, normalized, y, norm_weight,
                                      norm_bias, epsilon, fusion, e, writeback);
    }
    bool normalized_forward(Workspace &workspace, TensorView x,
                             TensorView normalized, TensorView y, TensorView norm_weight,
                             TensorView norm_bias, float epsilon, const ops::NormFusion &fusion,
                             ops::Epilogue e = {}, const QkvProjectionOutput *writeback = nullptr,
                             RowScaledView quantized_output = {}) const {
        validate_types(x, y);
        if (is_quantized()) {
            if (writeback)
                return false;
            if (ops::normalized_linear(x, weight_, bias_, normalized, y, norm_weight,
                                       norm_bias, epsilon, fusion, e, nullptr, quantized_output))
                return true;
            auto input = normalize_input(workspace, x, norm_weight, norm_bias, epsilon,
                                          fusion, normalized);
            ops::linear(input, weight_, bias_, y, e, {}, quantized_output);
            return true;
        }
        return !quantized_output.values.data &&
               normalized_forward(x, normalized, y, norm_weight, norm_bias, epsilon, fusion,
                                    e, writeback);
    }
    void forward(TensorView x, TensorView y, ops::Epilogue e = {}) const {
        apply(x, y, bias_, e, nullptr);
    }
    void forward(Workspace &workspace, TensorView x, TensorView y,
                 ops::Epilogue e = {}, RowScaledView quantized_output = {}) const {
        apply(x, y, bias_, e, &workspace, quantized_output);
    }

  private:
    void validate_types(TensorView x, TensorView y) const {
        if (x.type != activation_type_ ||
            (y.type != activation_type_ && (is_quantized() || y.type != Type::F32)))
            throw std::invalid_argument("linear activation dtype mismatch");
    }
    void apply(TensorView x, TensorView y, TensorView bias, ops::Epilogue e,
               Workspace *workspace, RowScaledView quantized_output = {}) const {
        const auto *quantized = std::get_if<RowScaledView>(&weight_);
        validate_types(x, y);
        RowScaledView scratch;
        if (quantized) {
            if (!workspace)
                throw std::logic_error("quantized linear requires an explicit workspace");
            scratch = workspace->view(x.rows, x.cols, quantized->values.type);
        }
        ops::linear(x, weight_, bias, y, e, scratch, quantized_output);
    }
};
} // namespace tts
