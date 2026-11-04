#include "layers/spectrogram.h"
#include <vector>
#include "ops/spectral.h"
#include "core/row_capacity.h"
#include "ops/basic.h"
#include "ops/storage.h"
#include <cmath>

namespace tts {
namespace {
double to_mel(double frequency_hz) {
    return frequency_hz < 1000 ? frequency_hz / (200. / 3.) : 15. + std::log(frequency_hz / 1000.) / (std::log(6.4) / 27.);
}
double to_hz(double mel) {
    return mel < 15 ? mel * (200. / 3.) : 1000. * std::exp((mel - 15.) * (std::log(6.4) / 27.));
}
} // namespace
Spectrogram::Spectrogram(SpectrogramConfig configuration, int max_frames, Device device, Type output_type)
    : config_(configuration), capacity_(max_frames), type_(output_type) {
    if (output_type != Type::F32 && output_type != Type::F16 && output_type != Type::BF16)
        throw std::invalid_argument("invalid spectrogram output type");
    if (configuration.fft < 2 || configuration.hop < 1 || configuration.window < 1 || configuration.window > configuration.fft || configuration.padding < 0 ||
        configuration.sample_rate < 1 || configuration.mel_channels < 0 || max_frames < 1 || configuration.min_frequency < 0 ||
        configuration.max_frequency < 0 || configuration.magnitude_epsilon < 0 || !std::isfinite(configuration.min_frequency) ||
        !std::isfinite(configuration.max_frequency) || !std::isfinite(configuration.magnitude_epsilon) ||
        !(configuration.log_floor > 0) || !std::isfinite(configuration.log_floor) || !std::isfinite(configuration.log_scale) ||
        !std::isfinite(configuration.log_bias))
        throw std::invalid_argument("invalid spectral configuration");
    std::vector<float> window(configuration.fft, 0);
    for (int coefficient_index = 0; coefficient_index < configuration.window; ++coefficient_index)
        window[(configuration.fft - configuration.window) / 2 + coefficient_index] = .5 - .5 * std::cos(2. * std::acos(-1.) * coefficient_index / configuration.window);
    window_ = Tensor({1, configuration.fft}, Type::F32, device);
    ops::upload_bytes(device.index(), window_.view().data, window.data(), window_.view().size());
    if (configuration.mel_channels) {
        double low = to_mel(configuration.min_frequency),
               high = to_mel(configuration.max_frequency ? configuration.max_frequency : configuration.sample_rate / 2.);
        if (!(high > low) || (configuration.max_frequency && configuration.max_frequency > configuration.sample_rate / 2.f))
            throw std::invalid_argument("invalid mel frequency range");
        std::vector<double> frequencies(configuration.mel_channels + 2);
        for (size_t coefficient_index = 0; coefficient_index < frequencies.size(); ++coefficient_index)
            frequencies[coefficient_index] = to_hz(low + (high - low) * coefficient_index / (configuration.mel_channels + 1));
        int bins = configuration.fft / 2 + 1;
        std::vector<float> coefficients(size_t(configuration.mel_channels) * bins);
        for (int mel_index = 0; mel_index < configuration.mel_channels; ++mel_index)
            for (int bin_index = 0; bin_index < bins; ++bin_index) {
                double frequency_hz = double(bin_index) * configuration.sample_rate / configuration.fft;
                float weight = std::max(
                    0., std::min((frequency_hz - frequencies[mel_index]) / (frequencies[mel_index + 1] - frequencies[mel_index]),
                                 (frequencies[mel_index + 2] - frequency_hz) /
                                     (frequencies[mel_index + 2] - frequencies[mel_index + 1])));
                coefficients[size_t(mel_index) * bins + bin_index] =
                    weight * (2. / (frequencies[mel_index + 2] - frequencies[mel_index]));
            }
        mel_ = Tensor({configuration.mel_channels, bins}, Type::F32, device);
        ops::upload_bytes(device.index(), mel_.view().data, coefficients.data(), mel_.view().size());
        ops::synchronize(device.index());
    }
    for (int64_t rows = 1; rows <= max_frames;) {
        int frame_capacity = std::min(max_frames, row_capacity(int(rows), 1));
        ops::prepare_fft(device.index(), configuration.fft, frame_capacity);
        rows = int64_t(frame_capacity) + 1;
    }
    ops::synchronize(device.index());
}
int Spectrogram::frames(int samples) const {
    return config_.frames(samples);
}
TensorView Spectrogram::forward(TensorView waveform, ForwardContext &context) const {
    const int batch = context.batch.batch(), samples = context.batch.time(), time = frames(samples);
    if (batch < 1 || int64_t(batch) * time > capacity_)
        throw std::invalid_argument("spectrogram exceeds initialized capacity");
    auto output = context.tensor({batch * time, output_channels()}, type_);
    Tensor::Scope scratch;
    Workspace workspace(*this);
    if (context.mode == ForwardContext::Prepare && output.data) prepare(workspace);
    else if (context.mode == ForwardContext::Execute) forward(workspace, waveform, batch, samples, output);
    return output;
}
int SpectrogramConfig::frames(int samples) const {
    int64_t padded_samples = int64_t(samples) + 2LL * padding - fft;
    if (fft < 2 || hop < 1 || padding < 0 || samples < 2 || padded_samples < 0 || padded_samples / hop >= INT32_MAX)
        throw std::invalid_argument("waveform is too short/long for the STFT");
    return int(padded_samples / hop + 1);
}
Spectrogram::Workspace::Workspace(const Spectrogram &spectrogram)
    : frames(size_t(spectrogram.capacity_) * spectrogram.config_.fft * sizeof(float)),
      spectrum(size_t(spectrogram.capacity_) * (spectrogram.config_.fft / 2 + 1) * 2 * sizeof(float)),
      magnitude(spectrogram.config_.mel_channels ? spectrum.size() / 2 : 0),
      mel(size_t(spectrogram.capacity_) * spectrogram.config_.mel_channels * sizeof(float)) {}
void Spectrogram::prepare(Workspace &workspace) const {
    if (!config_.mel_channels) return;
    const int bins = config_.fft / 2 + 1;
    ops::prepare_matmul(workspace.magnitude.view(1, bins, Type::F32), mel_.view(),
        workspace.mel.view(1, config_.mel_channels, Type::F32), true, 0, 1, 1, capacity_);
}
void Spectrogram::forward(Workspace &workspace, TensorView waveform, int batch, int samples,
                          TensorView output) const {
    int time = frames(samples);
    if (batch < 1 || int64_t(batch) * time > capacity_ || output.rows != int64_t(batch) * time ||
        output.cols != output_channels())
        throw std::invalid_argument("spectrogram exceeds initialized capacity");
    auto spectrum = workspace.spectrum.view(capacity_, (config_.fft / 2 + 1) * 2, Type::F32);
    ops::windowed_fft(waveform, window_.view(), workspace.frames.view(capacity_, config_.fft, Type::F32),
                       spectrum, batch, samples, time, config_.hop, config_.padding);
    if (!config_.mel_channels) {
        ops::spectrum_magnitude(spectrum, output, config_.magnitude_epsilon);
    } else {
        auto magnitude = workspace.magnitude.view(batch * time, config_.fft / 2 + 1, Type::F32);
        auto mel = workspace.mel.view(batch * time, config_.mel_channels, Type::F32);
        ops::spectrum_magnitude(spectrum, magnitude, config_.magnitude_epsilon);
        ops::matmul(magnitude, mel_.view(), mel, true);
        ops::spectral_log( mel, output, config_.log_floor, config_.log_scale, config_.log_bias);
    }
}
} // namespace tts
