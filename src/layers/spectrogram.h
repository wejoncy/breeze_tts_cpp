#pragma once
#include "ops/backend.h"
#include "runtime/tensor_storage.h"
#include <map>

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
    struct Plan;
    SpectrogramConfig config_;
    Tensor window_, mel_;
    std::map<int, std::unique_ptr<Plan>> plans_;
    Buffer fft_workspace_;
    int capacity_;

  public:
    struct Workspace {
        Buffer frames, spectrum, magnitude, mel;
        explicit Workspace(const Spectrogram &, BufferAllocator *);
    };
    Spectrogram(Backend &, SpectrogramConfig, int max_frames);
    ~Spectrogram();
    int frames(int samples) const;
    int output_channels() const {
        return config_.mel_channels ? config_.mel_channels : config_.fft / 2 + 1;
    }
    void forward(Workspace &, TensorView waveform, int batch, int samples,
                 TensorView output) const;
};
} // namespace tts
