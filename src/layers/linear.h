#pragma once
#include "core/buffer.h"
#include "ops/linear.h"
#include "layers/quantization.h"
#include <string>
#include <vector>

namespace tts {
class Weights;
struct ForwardContext;
class Linear {
    ops::LinearOperand weight_;
    TensorView bias_;
    Type activation_type_;

    TensorView matrix() const;
    Linear(ops::LinearOperand weight, TensorView bias, Type type);

  public:
    using Workspace = QuantizationWorkspace;
    Linear(Weights &weights, const std::string &name);
    Linear(Weights &weights, const std::string &name, bool quantize);
    Linear(TensorView weight, TensorView bias = {});
    Linear(RowScaledView weight, Type activation_type, TensorView bias = {});
    static Linear fused(Weights &weights, const std::string &derived,
                        const std::vector<std::string> &parts, TensorView bias, bool quantize);
    Type activation_type() const;
    TensorView forward(TensorView input, ForwardContext &context,
                       const ops::Epilogue &finish = {}) const;
    TensorView forward(TensorView input, TensorView output, ForwardContext &context,
                       const ops::Epilogue &finish = {}) const;
    TensorView project(TensorView input, TensorView output, ForwardContext &context) const;

  private:
    TensorView apply_context(TensorView input, TensorView output, ForwardContext &context,
                             TensorView bias, const ops::Epilogue &finish) const;

  public:
    TensorView forward_segments(TensorView input, TensorView segments, TensorView output,
                                ForwardContext &context) const;
    bool is_quantized() const;
    Workspace::Capacity workspace_capacity() const;
    TensorView dense_weight() const;
    TensorView bias() const;
    int input_channels() const;
    int output_channels() const;
    void prepare_range(int max_rows, ops::Activation activation = ops::Activation::Identity) const;
    void prepare_range(TensorView input, TensorView output, int max_rows,
                       ops::Activation activation = ops::Activation::Identity) const;
    void project(TensorView input, TensorView output) const;
    void project(Workspace &workspace, TensorView input, TensorView output) const;
    RowScaledView quantized_view(const Workspace &workspace, int rows) const;
    void project(RowScaledView input, TensorView output) const;
    RowScaledView normalize_input(Workspace &workspace, TensorView input, TensorView norm_weight,
                                  TensorView norm_bias, float epsilon,
                                  const ops::NormFusion &fusion, TensorView retained = {}) const;
    bool normalized_forward(TensorView input, TensorView normalized, TensorView output,
                            TensorView norm_weight, TensorView norm_bias, float epsilon,
                            const ops::NormFusion &fusion, ops::Epilogue epilogue = {},
                            const QkvProjectionOutput *writeback = nullptr) const;
    bool normalized_forward(Workspace &workspace, TensorView input, TensorView normalized,
                            TensorView output, TensorView norm_weight, TensorView norm_bias,
                            float epsilon, const ops::NormFusion &fusion,
                            ops::Epilogue epilogue = {},
                            const QkvProjectionOutput *writeback = nullptr,
                            RowScaledView quantized_output = {}) const;
    void forward(TensorView input, TensorView output, ops::Epilogue epilogue = {}) const;
    void forward(Workspace &workspace, TensorView input, TensorView output,
                 ops::Epilogue epilogue = {}, RowScaledView quantized_output = {}) const;

  private:
    QuantizationGroups quantization_layout() const;
    void validate_types(TensorView input, TensorView output) const;
    void apply(TensorView input, TensorView output, TensorView bias, ops::Epilogue epilogue,
               Workspace *workspace, RowScaledView quantized_output = {}) const;
};
} // namespace tts
