#pragma once
#include "core/buffer.h"
#include "core/tensor_storage.h"

namespace tts {
struct SpectrogramConfig {
    int fft, hop, window, padding, sample_rate;
    int mel_channels = 0;
    float min_frequency = 0, max_frequency = 0;
    float magnitude_epsilon = 1.e-8f, log_floor = 1.e-5f;
    float log_scale = 1, log_bias = 0;
    int frames(int samples) const;
};
// CPU-generated Hann/mel tables and all cuFFT plans are prepared at load. FFT
// scratch uses the same allocator as other operators; serving only selects a
// retained frame-capacity plan and supplies its initialized workspace.
class Spectrogram {
    SpectrogramConfig config_;
    Tensor window_, mel_;
    int capacity_;

  public:
    struct Workspace {
        Buffer frames, spectrum, magnitude, mel;
        explicit Workspace(const Spectrogram &);
    };
    Spectrogram(SpectrogramConfig, int max_frames, Device device);
    int frames(int samples) const;
    int output_channels() const {
        return config_.mel_channels ? config_.mel_channels : config_.fft / 2 + 1;
    }
    void prepare(Workspace &) const;
    void forward(Workspace &, TensorView waveform, int batch, int samples,
                 TensorView output) const;
};
} // namespace tts
