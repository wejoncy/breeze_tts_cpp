#include "layers/spectrogram.h"
#include "ops/spectral.h"
#include "core/row_capacity.h"
#include <cufft.h>
#include <cmath>

namespace tts {
namespace {
void fft_check(cufftResult result) {
    if (result != CUFFT_SUCCESS)
        throw DeviceError("cuFFT failed: " + std::to_string(int(result)));
}
double to_mel(double hz) {
    return hz < 1000 ? hz / (200. / 3.) : 15. + std::log(hz / 1000.) / (std::log(6.4) / 27.);
}
double to_hz(double mel) {
    return mel < 15 ? mel * (200. / 3.) : 1000. * std::exp((mel - 15.) * (std::log(6.4) / 27.));
}
} // namespace
struct Spectrogram::Plan {
    cufftHandle handle = 0;
    size_t workspace_bytes = 0;
    Plan(cudaStream_t stream, int fft, int rows) {
        fft_check(cufftCreate(&handle));
        try {
            fft_check(cufftSetAutoAllocation(handle, 0));
            fft_check(cufftMakePlanMany(handle, 1, &fft, nullptr, 1, fft, nullptr, 1, fft / 2 + 1,
                                        CUFFT_R2C, rows, &workspace_bytes));
            fft_check(cufftSetStream(handle, stream));
        } catch (...) {
            cufftDestroy(handle);
            throw;
        }
    }
    ~Plan() {
        if (handle)
            cufftDestroy(handle);
    }
};
Spectrogram::Spectrogram(Backend &gpu, SpectrogramConfig c, int max_frames)
    : config_(c), capacity_(max_frames) {
    if (c.fft < 2 || c.hop < 1 || c.window < 1 || c.window > c.fft || c.padding < 0 ||
        c.sample_rate < 1 || c.mel_channels < 0 || max_frames < 1 || c.min_frequency < 0 ||
        c.max_frequency < 0 || c.magnitude_epsilon < 0 || !std::isfinite(c.min_frequency) ||
        !std::isfinite(c.max_frequency) || !std::isfinite(c.magnitude_epsilon) ||
        !(c.log_floor > 0) || !std::isfinite(c.log_floor) || !std::isfinite(c.log_scale) ||
        !std::isfinite(c.log_bias))
        throw std::invalid_argument("invalid spectral configuration");
    std::vector<float> window(c.fft, 0);
    for (int i = 0; i < c.window; ++i)
        window[(c.fft - c.window) / 2 + i] = .5 - .5 * std::cos(2. * std::acos(-1.) * i / c.window);
    window_ = Tensor({1, c.fft}, Type::F32, &gpu.memory);
    gpu.upload(window_.view().data, window.data(), window_.view().size());
    if (c.mel_channels) {
        double low = to_mel(c.min_frequency),
               high = to_mel(c.max_frequency ? c.max_frequency : c.sample_rate / 2.);
        if (!(high > low) || (c.max_frequency && c.max_frequency > c.sample_rate / 2.f))
            throw std::invalid_argument("invalid mel frequency range");
        std::vector<double> frequencies(c.mel_channels + 2);
        for (size_t i = 0; i < frequencies.size(); ++i)
            frequencies[i] = to_hz(low + (high - low) * i / (c.mel_channels + 1));
        int bins = c.fft / 2 + 1;
        std::vector<float> coefficients(size_t(c.mel_channels) * bins);
        for (int m = 0; m < c.mel_channels; ++m)
            for (int k = 0; k < bins; ++k) {
                double hz = double(k) * c.sample_rate / c.fft;
                float weight = std::max(
                    0., std::min((hz - frequencies[m]) / (frequencies[m + 1] - frequencies[m]),
                                 (frequencies[m + 2] - hz) /
                                     (frequencies[m + 2] - frequencies[m + 1])));
                coefficients[size_t(m) * bins + k] =
                    weight * (2. / (frequencies[m + 2] - frequencies[m]));
            }
        mel_ = Tensor({c.mel_channels, bins}, Type::F32, &gpu.memory);
        gpu.upload(mel_.view().data, coefficients.data(), mel_.view().size());
        gpu.sync();
    }
    size_t workspace_bytes = 0;
    for (int64_t rows = 1; rows <= max_frames;) {
        int cap = std::min(max_frames, row_capacity(int(rows), 1));
        auto plan = std::make_unique<Plan>(gpu.stream, c.fft, cap);
        workspace_bytes = std::max(workspace_bytes, plan->workspace_bytes);
        plans_.emplace(cap, std::move(plan));
        rows = int64_t(cap) + 1;
    }
    fft_workspace_ = Buffer(workspace_bytes, &gpu.memory);
    for (const auto &[rows, plan] : plans_)
        fft_check(cufftSetWorkArea(plan->handle, fft_workspace_.data()));
    gpu.sync();
}
Spectrogram::~Spectrogram() = default;
int Spectrogram::frames(int samples) const {
    return config_.frames(samples);
}
int SpectrogramConfig::frames(int samples) const {
    int64_t n = int64_t(samples) + 2LL * padding - fft;
    if (fft < 2 || hop < 1 || padding < 0 || samples < 2 || n < 0 || n / hop >= INT32_MAX)
        throw std::invalid_argument("waveform is too short/long for the STFT");
    return int(n / hop + 1);
}
Spectrogram::Workspace::Workspace(const Spectrogram &s, BufferAllocator *storage)
    : frames(size_t(s.capacity_) * s.config_.fft * sizeof(float), storage),
      spectrum(size_t(s.capacity_) * (s.config_.fft / 2 + 1) * 2 * sizeof(float), storage),
      magnitude(s.config_.mel_channels ? spectrum.size() / 2 : 0, storage),
      mel(size_t(s.capacity_) * s.config_.mel_channels * sizeof(float), storage) {}
void Spectrogram::forward(Workspace &w, TensorView waveform, int batch, int samples,
                          TensorView out) const {
    auto &gpu = backend_for(waveform);
    int time = frames(samples);
    if (batch < 1 || int64_t(batch) * time > capacity_ || out.rows != int64_t(batch) * time ||
        out.cols != output_channels())
        throw std::invalid_argument("spectrogram exceeds initialized capacity");
    auto plan = plans_.lower_bound(batch * time);
    if (plan == plans_.end())
        throw std::logic_error("missing initialized FFT plan");
    stft_window(gpu.stream, waveform, window_.view(),
                w.frames.view(plan->first, config_.fft, Type::F32), batch, samples, time,
                config_.hop, config_.padding);
    fft_check(cufftExecR2C(plan->second->handle, w.frames.ptr<float>(),
                           static_cast<cufftComplex *>(w.spectrum.data())));
    if (!config_.mel_channels) {
        spectrum_magnitude(gpu.stream, w.spectrum.data(), out, config_.magnitude_epsilon);
    } else {
        auto magnitude = w.magnitude.view(batch * time, config_.fft / 2 + 1, Type::F32);
        auto mel = w.mel.view(batch * time, config_.mel_channels, Type::F32);
        spectrum_magnitude(gpu.stream, w.spectrum.data(), magnitude, config_.magnitude_epsilon);
        gpu.linear(magnitude, mel_.view(), mel, true);
        spectral_log(gpu.stream, mel, out, config_.log_floor, config_.log_scale, config_.log_bias);
    }
}
} // namespace tts
