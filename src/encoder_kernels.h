#pragma once
#include "tensor.h"
#include <cuda_runtime_api.h>
namespace tts {
void encoder_first_conv(cudaStream_t, TensorView x, TensorView weight, TensorView bias, TensorView out,
                        int kernel);
void encoder_conv_finish(cudaStream_t, TensorView x, TensorView projected, TensorView weight,
                         TensorView bias, TensorView out, int kernel, int stride, bool replicate,
                         bool elu, TensorView residual = {});
void encoder_quantize(cudaStream_t, TensorView residual, TensorView scores, TensorView table,
                      TensorView norms, int *codes, int codebook);
} // namespace tts
