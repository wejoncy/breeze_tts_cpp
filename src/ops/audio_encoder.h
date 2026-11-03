#pragma once
#include "core/tensor.h"
#include <cuda_runtime_api.h>
namespace tts {
void encoder_first_conv(cudaStream_t, TensorView x, TensorView weight, TensorView bias, TensorView out,
                        int kernel);
void encoder_conv_finish(cudaStream_t, TensorView x, TensorView projected, TensorView weight,
                         TensorView bias, TensorView out, int kernel, int stride, bool replicate,
                         bool elu, TensorView residual = {}, int dilation = 1);
} // namespace tts
