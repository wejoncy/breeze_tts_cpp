#pragma once
#include <memory>
#include <string>

namespace tts {
// Request-owned streaming PCM16 resampling. A final call drains the filter tail.
class AudioSpeed {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    AudioSpeed(int sample_rate, double speed);
    ~AudioSpeed();
    std::string process(std::string pcm, bool final);
};
} // namespace tts
