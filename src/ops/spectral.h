#pragma once
#include "core/tensor.h"
#include <cuda_runtime_api.h>

namespace tts {
void stft_window(cudaStream_t, TensorView waveform, TensorView window, TensorView frames,
                 int batch, int samples, int time, int hop, int padding);
void spectrum_magnitude(cudaStream_t, const void *complex, TensorView out, float epsilon);
void spectral_log(cudaStream_t, TensorView input, TensorView output, float floor,
                  float scale, float bias);
} // namespace tts
