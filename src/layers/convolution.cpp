#include "ops/basic.h"
#include "layers/convolution.h"
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
} // namespace tts
