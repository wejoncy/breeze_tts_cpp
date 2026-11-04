#include "ops/basic.h"
#include "layers/convolution.h"
#include <cstring>

namespace tts {
Convolution::Convolution(Weights &weights, const std::string &prefix, ops::ConvGeometry geometry)
    : Convolution(weights, prefix, geometry, weights.get(prefix + ".bias")) {}
Convolution::Convolution(Weights &weights, const std::string &prefix, ops::ConvGeometry geometry,
                         TensorView bias)
    : geometry_(geometry), bias_(bias) {
    geometry_.output_length(1 + geometry.kernel * geometry.dilation);
    const auto &entry = weights.entry(prefix + ".weight");
    int in = geometry.input_channels / geometry.groups;
    int out = geometry.output_channels / geometry.groups;
    std::vector<int> expected =
        geometry.transpose ? std::vector<int>{geometry.input_channels, out, geometry.kernel}
                           : std::vector<int>{geometry.output_channels, in, geometry.kernel};
    if (entry.shape != expected)
        throw std::invalid_argument("convolution checkpoint/configuration mismatch: " + prefix);
    std::vector<char> packed(entry.size);
    size_t element_bytes = bytes(entry.type);
    for (int group = 0; group < geometry.groups; ++group)
        for (int tap = 0; tap < geometry.kernel; ++tap)
            for (int o = 0; o < out; ++o)
                for (int i = 0; i < in; ++i) {
                    size_t source =
                        geometry.transpose
                            ? (size_t(group * in + i) * out + o) * geometry.kernel + tap
                            : (size_t(group * out + o) * in + i) * geometry.kernel + tap;
                    size_t dest = ((size_t(group) * geometry.kernel + tap) * out + o) * in + i;
                    std::memcpy(packed.data() + dest * element_bytes,
                                static_cast<const char *>(entry.data) + source * element_bytes,
                                element_bytes);
                }
    weight_ = weights.custom(prefix + ".packed", packed.data(),
                             geometry.kernel * geometry.output_channels, in, entry.type);
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
