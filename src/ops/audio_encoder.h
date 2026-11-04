#pragma once
#include "core/tensor.h"
namespace tts {
void encoder_first_conv( TensorView x, TensorView weight, TensorView bias, TensorView out,
                        int kernel);
void encoder_conv_finish( TensorView x, TensorView projected, TensorView weight,
                         TensorView bias, TensorView out, int kernel, int stride, bool replicate,
                         bool elu, TensorView residual = {}, int dilation = 1);
} // namespace tts
