#include "ops/convolution.h"
#include "ops/basic.h"
#include "runtime/cuda_context.h"

namespace tts::ops {
size_t convolution_workspace_bytes(const ConvGeometry &g, int batch, int time) {
    if (batch < 1 || time < 1)
        throw std::invalid_argument("invalid convolution input");
    if (g.input_channels / g.groups <= 4)
        return 0;
    size_t rows = size_t(batch) * time;
    size_t width = size_t(g.kernel) * g.output_channels;
    if (width > SIZE_MAX / sizeof(float) || rows > SIZE_MAX / (width * sizeof(float)))
        throw std::overflow_error("convolution workspace size");
    return rows * width * sizeof(float);
}
void convolution(TensorView x, TensorView weight, TensorView bias, TensorView y,
                 int batch, int time, const ConvGeometry &g, TensorView workspace, Epilogue e,
                 const SequenceLayout *input, const SequenceLayout *output) {
    auto &gpu = execution_context(x, {weight, bias, y, workspace, e.residual, e.condition, e.gate});
    if (bool(input) != bool(output)) throw std::invalid_argument("incomplete packed convolution");
    if (batch < 1 || (input ? input->rows : int64_t(batch) * time) != x.rows || x.cols != g.input_channels ||
        x.type != weight.type || y.type != weight.type ||
        (output ? output->rows : int64_t(batch) * g.output_length(time)) != y.rows || y.cols != g.output_channels ||
        e.bias.data)
        throw std::invalid_argument("convolution input/output mismatch");
    e.bias = bias;
    if (!convolution_workspace_bytes(g, batch, time)) {
        direct_convolution( x, weight, y, batch, time, g, e, input, output);
        return;
    }
    if (workspace.type != Type::F32 || workspace.rows != x.rows ||
        workspace.cols != g.kernel * g.output_channels)
        throw std::invalid_argument("convolution projection workspace mismatch");
    tts::ops::grouped_matmul(x, weight, workspace, g.groups);
    convolution_finish( workspace, y, batch, time, g, e, input, output);
}
} // namespace tts::ops
