#pragma once
#include "core/quantization.h"
#include <cuda_runtime_api.h>
namespace tts {
// Initialization-only query of the compiled kernel image and device capability.
bool quant_gated_supports_clusters(int device);
void launch_quantized_gated(TensorView input, RowScaledView output, int kind, cudaStream_t stream,
                            int multiprocessors, bool clusters);
} // namespace tts
