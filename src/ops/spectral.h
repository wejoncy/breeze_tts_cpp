#pragma once
#include "core/tensor.h"

namespace tts {
namespace ops {
void prepare_fft(int device, int fft_size, int rows);
void windowed_fft(TensorView waveform, TensorView window, TensorView frames, TensorView spectrum,
                   int batch, int samples, int time, int hop, int padding);
void spectrum_magnitude(TensorView spectrum, TensorView output, float epsilon);
void spectral_log(TensorView input, TensorView output, float floor, float scale, float bias);
}
void stft_window( TensorView waveform, TensorView window, TensorView frames,
                 int batch, int samples, int time, int hop, int padding);
void spectrum_magnitude( const void *complex, TensorView out, float epsilon);
void spectral_log( TensorView input, TensorView output, float floor,
                  float scale, float bias);
} // namespace tts
