#pragma once
#include "ops/sequence.h"
#include "core/quantization.h"
#include <variant>

namespace tts {
class Backend;
namespace ops {
using LinearOperand = std::variant<TensorView, RowScaledView>;
// With quantized_output, output is disposable projection scratch; the result is in
// quantized_output. Use Epilogue::raw to retain pre-activation values explicitly.
void linear(const LinearOperand &input, const LinearOperand &weight, TensorView bias,
            TensorView output, Epilogue = {}, RowScaledView scratch = {},
            RowScaledView quantized_output = {});
bool normalized_linear(TensorView input, const LinearOperand &weight, TensorView bias,
                       TensorView normalized, TensorView output, TensorView norm_weight,
                       TensorView norm_bias, float epsilon, const NormFusion &, Epilogue = {},
                       const QkvProjectionOutput *writeback = nullptr,
                       RowScaledView quantized_output = {});
} // namespace ops
} // namespace tts
