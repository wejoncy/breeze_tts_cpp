#pragma once
#include "core/tensor.h"
#include <cuda_runtime_api.h>

namespace tts::ops {
// FP32 projections, FP16/BF16 biases, and FP32 descriptor tables are consumed
// directly. Legal labels use a character-major bitset of 32 labels per word.
void select_pronunciation(cudaStream_t, TensorView logits, TensorView pos_logits,
                          TensorView bias, TensorView pos_bias, TensorView descriptor_bias,
                          TensorView char_descriptor, TensorView second_order_descriptor,
                          TensorView legal, const int *query_positions, const int *char_ids,
                          int *output_ids);
} // namespace tts::ops
