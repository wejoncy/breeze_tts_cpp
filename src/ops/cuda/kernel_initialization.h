#pragma once
#include "runtime/device.h"

namespace tts::cuda_kernels {
// Init-only function resolution; no kernel launch or shape-specific work.
template <class Kernel> void materialize(Kernel kernel) {
    cudaFuncAttributes attributes{};
    CUDA(cudaFuncGetAttributes(&attributes, kernel));
}
void initialize_pronunciation();
void initialize_vector_quantization();
void initialize_spectral();
void initialize_relative_attention();
void initialize_audio();
} // namespace tts::cuda_kernels
