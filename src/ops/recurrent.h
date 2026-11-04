#pragma once
#include "core/tensor.h"
namespace tts::ops {
// Small CPU frontend networks use the same typed tensor contract as GPU ops.
void gather_rows(TensorView table, TensorView indices, TensorView output);
void gru_cell(TensorView input, TensorView input_weight, TensorView recurrent_weight,
              TensorView input_bias, TensorView recurrent_bias, TensorView state,
              TensorView workspace, TensorView active = {});
void project_argmax(TensorView input, TensorView weight, TensorView bias, TensorView indices);
} // namespace tts::ops
