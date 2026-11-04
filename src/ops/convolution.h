#pragma once
#include "ops/sequence.h"

namespace tts {
namespace ops {
// Half-open input interval sufficient for a requested convolution output.
inline std::pair<int64_t, int64_t> convolution_input_span(
    const ConvGeometry &g, std::pair<int64_t, int64_t> output) {
    auto floor_div = [](int64_t a, int64_t b) { return a / b - (a % b < 0); };
    int64_t extent = int64_t(g.dilation) * (g.kernel - 1);
    return g.transpose
               ? std::pair<int64_t, int64_t>{
                     -floor_div(-(output.first + g.padding - extent), g.stride),
                     floor_div(output.second - 1 + g.padding, g.stride) + 1}
               : std::pair<int64_t, int64_t>{output.first * g.stride - g.padding,
                     (output.second - 1) * g.stride - g.padding + extent + 1};
}
size_t convolution_workspace_bytes(const ConvGeometry &, int batch, int input_length);
void convolution(TensorView input, TensorView weight, TensorView bias,
                 TensorView output, int batch, int input_length, const ConvGeometry &,
                 TensorView workspace, Epilogue = {}, const SequenceLayout *input_layout = nullptr,
                 const SequenceLayout *output_layout = nullptr);
} // namespace ops
} // namespace tts
