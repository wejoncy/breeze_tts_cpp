#pragma once
#include "core/quantization.h"
#include <cuda_runtime_api.h>

namespace tts {
void launch_row_quantize(TensorView source, RowScaledView destination, cudaStream_t stream);
} // namespace tts
