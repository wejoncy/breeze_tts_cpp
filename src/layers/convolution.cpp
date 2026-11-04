#include "ops/basic.h"
#include "layers/convolution.h"
#include "ops/audio_encoder.h"
#include <cstring>

namespace tts {
TensorView load_convolution_weight(Weights &weights, const std::string &name, bool transpose,
                                   int groups, int phases) {
    const auto &entry = weights.entry(name + ".weight");
    const auto &shape = entry.shape;
    if (shape.size() != 3 || groups < 1 || phases < 1 || shape[0] < 1 || shape[1] < 1 ||
        shape[2] < 1 || shape[0] % groups || shape[2] % phases)
        throw std::invalid_argument("invalid convolution weight layout: " + name);
    const int64_t total_output_channels = transpose ? int64_t(shape[1]) * groups : shape[0];
    if (total_output_channels > INT32_MAX / shape[2])
        throw std::invalid_argument("convolution weight exceeds tensor dimensions: " + name);
    const int input_channels = transpose ? shape[0] / groups : shape[1];
    const int output_channels = transpose ? shape[1] : shape[0] / groups;
    const int kernel = shape[2], taps = kernel / phases;
    const size_t element_bytes = bytes(entry.type);
    std::vector<char> packed(entry.size);
    for (int group = 0; group < groups; ++group)
        for (int packed_tap = 0; packed_tap < kernel; ++packed_tap) {
            const int source_tap = (packed_tap % taps) * phases + packed_tap / taps;
            for (int output_channel = 0; output_channel < output_channels; ++output_channel)
                for (int input_channel = 0; input_channel < input_channels; ++input_channel) {
                    const size_t source_channel = transpose
                        ? size_t(group * input_channels + input_channel) * output_channels + output_channel
                        : size_t(group * output_channels + output_channel) * input_channels + input_channel;
                    const size_t source = source_channel * kernel + source_tap;
                    const size_t destination = ((size_t(group) * kernel + packed_tap) *
                        output_channels + output_channel) * input_channels + input_channel;
                    std::memcpy(packed.data() + destination * element_bytes,
                        static_cast<const char *>(entry.data) + source * element_bytes, element_bytes);
                }
        }
    return weights.custom(name + ".packed", packed.data(), kernel * output_channels * groups,
                          input_channels, entry.type);
}
Convolution::Convolution(Weights &weights, const std::string &prefix, ops::ConvGeometry geometry)
    : Convolution(weights, prefix, geometry, weights.get(prefix + ".bias")) {}
Convolution::Convolution(Weights &weights, const std::string &prefix, ops::ConvGeometry geometry,
                         TensorView bias)
    : geometry_(geometry), bias_(bias) {
    geometry_.output_length(1 + geometry.kernel * geometry.dilation);
    const auto &entry = weights.entry(prefix + ".weight");
    const int input_channels = geometry.input_channels / geometry.groups;
    const int output_channels = geometry.output_channels / geometry.groups;
    std::vector<int> expected =
        geometry.transpose ? std::vector<int>{geometry.input_channels, output_channels, geometry.kernel}
                           : std::vector<int>{geometry.output_channels, input_channels, geometry.kernel};
    if (entry.shape != expected)
        throw std::invalid_argument("convolution checkpoint/configuration mismatch: " + prefix);
    weight_ = load_convolution_weight(weights, prefix, geometry.transpose, geometry.groups);
    if (bias_.data && (bias_.numel() != size_t(geometry.output_channels) || bias_.type != type()))
        throw std::invalid_argument("convolution bias mismatch: " + prefix);
}
size_t Convolution::workspace_bytes(int batch, int input_length) const {
    return ops::convolution_workspace_bytes(geometry_, batch, input_length);
}
void Convolution::prepare(TensorView input, TensorView projection,
                          int max_rows) const {

    if (workspace_bytes(1, 1))
        ops::prepare_matmul(input, weight_, projection, false, 0, 1, geometry_.groups, max_rows);
}
void Convolution::forward(TensorView x, TensorView y, const ForwardBatch &batch,
                          TensorView workspace, ops::Epilogue e, const SequenceLayout *output) const {
    auto *input = batch.sequence.addressing();
    if (input && !output && x.rows == y.rows &&
        geometry_.output_length(batch.time()) == batch.time())
        output = input;
    ops::convolution(x, weight_, bias_, y, batch.batch(), batch.time(), geometry_, workspace,
                     e, input, output);
}
CausalConvolution::CausalConvolution(Weights &weights, const std::string &name, Config config)
    : CausalConvolution(weights, name, config, weights.get(name + ".bias")) {}
CausalConvolution::CausalConvolution(Weights &weights, const std::string &name, Config config,
                                     TensorView bias)
    : bias(bias), stride(config.stride), dilation(config.dilation), transpose(config.transpose),
      depthwise(config.depthwise), replicate(config.replicate) {
    const auto &entry = weights.entry(name + ".weight");
    if (entry.type != Type::F32 || entry.shape.size() != 3 || entry.shape[0] < 1 ||
        entry.shape[1] < 1 || entry.shape[2] < 1 || stride < 1 || dilation < 1 ||
        (depthwise && (transpose || stride != 1 || entry.shape[1] != 1)) ||
        (transpose && (entry.shape[2] < stride || replicate)) ||
        (!transpose && (entry.shape[2] % stride || (stride > 1 && dilation > 1))))
        throw std::invalid_argument("unsupported causal convolution geometry/dtype: " + name);
    kernel = entry.shape[2];
    in = transpose ? entry.shape[0] : entry.shape[1];
    out = transpose ? entry.shape[1] : entry.shape[0];
    if (depthwise) in = out;
    if (kernel > INT32_MAX / out || int64_t(kernel - 1) * dilation >= INT32_MAX)
        throw std::invalid_argument("causal convolution indexing overflow: " + name);
    if (bias.data && (bias.type != Type::F32 || bias.numel() != size_t(out)))
        throw std::invalid_argument("causal convolution bias shape: " + name);
    weight = depthwise ? weights.get(name + ".weight")
                       : load_convolution_weight(weights, name, transpose, 1, transpose ? 1 : stride);
}
void CausalConvolution::project(TensorView input, TensorView projection, int max_input_rows) const {
    const int input_rows = max_input_rows ? max_input_rows : input.rows, taps = kernel / stride;
    for (int phase = 0; phase < stride && phase < input_rows; ++phase) {
        const int count = (input_rows - 1 - phase) / stride + 1;
        auto source = input.every_rows(phase, stride, max_input_rows ? 1 : count);
        auto target = projection.columns(phase * taps * out, taps * out).slice(0, source.rows);
        auto matrix = weight.slice(phase * taps * out, taps * out);
        if (max_input_rows) ops::prepare_matmul(source, matrix, target, true, 0, 1, 1, count);
        else ops::matmul(source, matrix, target, true);
    }
}
void CausalConvolution::prepare(TensorView input, TensorView projection, int max_input_rows) const {
    if (transpose || depthwise || max_input_rows < 1)
        throw std::invalid_argument("invalid causal sequence projection preparation");
    project(input, projection, max_input_rows);
}
void CausalConvolution::forward(TensorView input, TensorView output, TensorView projection,
                                 bool elu, TensorView residual) const {
    if (transpose || depthwise || input.type != Type::F32 || output.type != Type::F32 ||
        input.cols != in || output.rows != rows(input.rows) || output.cols != out)
        throw std::invalid_argument("causal convolution input/output shape");
    if (in == 1 && stride == 1 && dilation == 1 && !replicate && !elu && !residual.numel()) {
        encoder_first_conv(input, weight, bias, output, kernel);
        return;
    }
    auto projected = projection.reshape({int(projection.numel() / (kernel * out)), kernel * out})
                               .slice(0, output.rows);
    project(input, projected);
    encoder_conv_finish(input, projected, weight, bias, output, kernel, stride, replicate, elu, residual, dilation);
}
void CausalConvolution::forward(TensorView input, TensorView output, TensorView projection,
                                 const ForwardBatch &batch, const ConvHistory *histories) const {
    if (!histories || replicate || (!transpose && stride != 1))
        throw std::invalid_argument("causal convolution history/stride mismatch");
    if (input.type != Type::F32 || output.type != Type::F32 || input.rows != batch.rows() ||
        input.cols != in || output.rows != int64_t(batch.batch()) * rows(batch.time()) || output.cols != out)
        throw std::invalid_argument("causal convolution input/output shape");
    if (depthwise) {
        tts::depthwise(input, {}, weight, bias, output, batch.batch(), batch.time(), kernel, nullptr, histories);
    } else {
        ops::matmul(input, weight, projection);
        conv_accumulate(projection, {}, bias, output, batch.batch(), batch.time(), kernel,
                        dilation, stride, transpose, nullptr, histories);
    }
}
} // namespace tts
