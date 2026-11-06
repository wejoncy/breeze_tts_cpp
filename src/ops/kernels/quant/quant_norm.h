#pragma once
#include "core/quantization.h"
#include <cuda_runtime_api.h>
namespace tts {
void launch_quantized_norm(TensorView residual, TensorView update, TensorView weight,
                           RowScaledView output, float epsilon, bool gemma, TensorView update_scale,
                           cudaStream_t stream);
} // namespace tts
