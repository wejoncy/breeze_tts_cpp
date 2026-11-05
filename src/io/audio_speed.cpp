#include "io/audio_speed.h"
#include <soxr.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace tts {
struct AudioSpeed::Impl {
    soxr_t resampler = nullptr;
    bool finished = false;
    ~Impl() {
        if (resampler)
            soxr_delete(resampler);
    }
};
AudioSpeed::AudioSpeed(int sample_rate, double speed) : impl_(std::make_unique<Impl>()) {
    if (sample_rate <= 0 || !std::isfinite(speed) || speed < .25 || speed > 4)
        throw std::invalid_argument("invalid PCM speed");
    if (speed == 1)
        return;
    auto io = soxr_io_spec(SOXR_INT16_I, SOXR_INT16_I);
    auto quality = soxr_quality_spec(SOXR_HQ, 0);
    soxr_error_t error = nullptr;
    impl_->resampler =
        soxr_create(sample_rate * speed, sample_rate, 1, &error, &io, &quality, nullptr);
    if (error || !impl_->resampler)
        throw std::runtime_error(error ? error : "resampler creation failed");
}
AudioSpeed::~AudioSpeed() = default;
std::string AudioSpeed::process(std::string pcm, bool final) {
    if (impl_->finished || pcm.size() % sizeof(int16_t))
        throw std::logic_error("invalid PCM speed stream state");
    impl_->finished = final;
    if (!impl_->resampler)
        return pcm;
    std::string output;
    std::array<int16_t, 4096> buffer;
    const size_t samples = pcm.size() / sizeof(int16_t);
    size_t position = 0;
    auto process = [&](bool drain) {
        size_t used = 0, written = 0;
        auto error = soxr_process(
            impl_->resampler, drain ? nullptr : pcm.data() + position * sizeof(int16_t),
            samples - position, &used, buffer.data(), buffer.size(), &written);
        if (error)
            throw std::runtime_error(error);
        position += used;
        output.append(reinterpret_cast<const char *>(buffer.data()), written * sizeof(int16_t));
        if (!drain && !used && !written)
            throw std::runtime_error("resampler made no progress");
        return written;
    };
    while (position < samples)
        process(false);
    if (final)
        while (process(true)) {
        }
    return output;
}
} // namespace tts
