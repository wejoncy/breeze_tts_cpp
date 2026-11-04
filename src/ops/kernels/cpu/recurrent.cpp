#include "ops/recurrent.h"
#include <cmath>
#include <limits>

namespace tts::ops {
namespace {
bool host(TensorView tensor, Type type) {
    return tensor.device < 0 && tensor.type == type && tensor.data;
}
float dot(const float *input, const float *weight, int channels) {
    float sum = 0;
    for (int channel = 0; channel < channels; ++channel) sum += input[channel] * weight[channel];
    return sum;
}
} // namespace
void gather_rows(TensorView table, TensorView indices, TensorView output) {
    if (!host(table, Type::F32) || !host(output, Type::F32) || !host(indices, Type::I32) ||
        indices.cols != 1 || output.rows != indices.rows || output.cols != table.cols || table.overlaps(output))
        throw std::invalid_argument("invalid CPU embedding tensors");
    for (int row = 0; row < output.rows; ++row) {
        const int index = indices.ptr<int>()[size_t(row) * indices.ld()];
        if (index < 0 || index >= table.rows) throw std::out_of_range("embedding index");
        for (int channel = 0; channel < output.cols; ++channel)
            output.ptr<float>()[size_t(row) * output.ld() + channel] = table.ptr<float>()[size_t(index) * table.ld() + channel];
    }
}
void gru_cell(TensorView input, TensorView input_weight, TensorView recurrent_weight,
              TensorView input_bias, TensorView recurrent_bias, TensorView state,
              TensorView workspace, TensorView active) {
    const int hidden = state.cols;
    for (auto tensor : {input, input_weight, recurrent_weight, input_bias, recurrent_bias, state, workspace})
        if (!host(tensor, Type::F32)) throw std::invalid_argument("GRU requires CPU float tensors");
    if (hidden < 1 || hidden > INT32_MAX / 6 || input.rows != state.rows || workspace.rows != state.rows ||
        workspace.cols < 6 * hidden || input_weight.rows != 3 * hidden || input_weight.cols != input.cols ||
        recurrent_weight.rows != 3 * hidden || recurrent_weight.cols != hidden ||
        input_bias.numel() != size_t(3 * hidden) || recurrent_bias.numel() != input_bias.numel() ||
        !input_bias.dense() || !recurrent_bias.dense() ||
        (active.data && (!host(active, Type::I32) || active.rows != state.rows || active.cols != 1)))
        throw std::invalid_argument("GRU tensor geometry mismatch");
    for (int row = 0; row < state.rows; ++row) {
        if (active.data && !active.ptr<int>()[size_t(row) * active.ld()]) continue;
        const auto *values = input.ptr<float>() + size_t(row) * input.ld();
        auto *hidden_state = state.ptr<float>() + size_t(row) * state.ld();
        auto *projected = workspace.ptr<float>() + size_t(row) * workspace.ld();
        auto *recurrent = projected + 3 * hidden;
        for (int channel = 0; channel < 3 * hidden; ++channel) {
            projected[channel] = dot(values, input_weight.ptr<float>() + size_t(channel) * input_weight.ld(), input.cols) + input_bias.ptr<float>()[channel];
            recurrent[channel] = dot(hidden_state, recurrent_weight.ptr<float>() + size_t(channel) * recurrent_weight.ld(), hidden) + recurrent_bias.ptr<float>()[channel];
        }
        for (int channel = 0; channel < hidden; ++channel) {
            const float reset = 1.f / (1.f + std::exp(-(projected[channel] + recurrent[channel])));
            const float update = 1.f / (1.f + std::exp(-(projected[hidden + channel] + recurrent[hidden + channel])));
            const float candidate = std::tanh(projected[2 * hidden + channel] + reset * recurrent[2 * hidden + channel]);
            hidden_state[channel] = (1.f - update) * candidate + update * hidden_state[channel];
        }
    }
}
void project_argmax(TensorView input, TensorView weight, TensorView bias, TensorView indices) {
    if (!host(input, Type::F32) || !host(weight, Type::F32) || !host(bias, Type::F32) ||
        !host(indices, Type::I32) || weight.cols != input.cols || weight.rows < 1 ||
        bias.numel() != size_t(weight.rows) || !bias.dense() || indices.rows != input.rows || indices.cols != 1)
        throw std::invalid_argument("invalid CPU classification tensors");
    for (int row = 0; row < input.rows; ++row) {
        float maximum = -std::numeric_limits<float>::infinity();
        int selected = 0;
        for (int channel = 0; channel < weight.rows; ++channel) {
            const float value = dot(input.ptr<float>() + size_t(row) * input.ld(),
                weight.ptr<float>() + size_t(channel) * weight.ld(), input.cols) + bias.ptr<float>()[channel];
            if (value > maximum) { selected = channel; maximum = value; }
        }
        indices.ptr<int>()[size_t(row) * indices.ld()] = selected;
    }
}
} // namespace tts::ops
