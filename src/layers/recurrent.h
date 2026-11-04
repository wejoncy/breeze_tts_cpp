#pragma once
#include "ops/recurrent.h"

namespace tts {
class GRUCell {
    TensorView input_, recurrent_, input_bias_, recurrent_bias_;
  public:
    GRUCell(TensorView input, TensorView recurrent, TensorView input_bias, TensorView recurrent_bias)
        : input_(input), recurrent_(recurrent), input_bias_(input_bias), recurrent_bias_(recurrent_bias) {
        if (recurrent.cols < 1 || recurrent.cols > INT32_MAX / 6 || input.rows != 3 * recurrent.cols ||
            recurrent.rows != input.rows || input_bias.numel() != size_t(input.rows) || recurrent_bias.numel() != input_bias.numel())
            throw std::invalid_argument("disconnected GRU weights");
    }
    int hidden() const { return recurrent_.cols; }
    int input_channels() const { return input_.cols; }
    void forward(TensorView input, TensorView state, TensorView workspace, TensorView active = {}) const {
        ops::gru_cell(input, input_, recurrent_, input_bias_, recurrent_bias_, state, workspace, active);
    }
};
} // namespace tts
