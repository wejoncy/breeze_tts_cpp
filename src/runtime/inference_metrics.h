#pragma once
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>

namespace tts {
struct CacheUsage {
    size_t page_bytes = 0, device_bytes = 0, state_bytes = 0;
    int used_pages = 0, backed_pages = 0;
    size_t active_rows = 0, state_rows = 0;
};
struct SchedulerStats {
    size_t running_requests = 0, waiting_requests = 0;
};
struct IterationStats {
    // Actual mono PCM appended to completed outputs, after overlap trimming.
    uint64_t audio_samples = 0;
    CacheUsage cache;
};
// Engine-thread aggregate counters. No request identities, callbacks or transport objects.
class InferenceMetrics {
  public:
    using Clock = std::chrono::steady_clock;

  private:
    Clock::time_point last_log_;
    const int sample_rate_;
    uint64_t audio_samples_ = 0;
    double request_rtf_sum_ = 0;
    size_t completed_requests_ = 0;
    size_t peak_bytes_ = 0;
    size_t last_active_rows_ = 0;
    bool last_activity_ = false;

  public:
    explicit InferenceMetrics(int sample_rate, Clock::time_point now = Clock::now())
        : last_log_(now), sample_rate_(sample_rate) {}
    Clock::time_point next_log() const { return last_log_ + std::chrono::seconds(1); }
    bool should_log(const CacheUsage &cache, Clock::time_point now) const {
        return now >= next_log() || cache.active_rows != last_active_rows_;
    }
    void record(const IterationStats &stats) {
        audio_samples_ += stats.audio_samples;
        const auto &cache = stats.cache;
        peak_bytes_ = std::max(peak_bytes_, size_t(cache.used_pages) * cache.page_bytes +
                              cache.active_rows * cache.state_bytes);
    }
    void finish_request(double elapsed_seconds, uint64_t audio_samples) {
        // A request can end at EOS without emitting audio; its RTF is undefined.
        if (!audio_samples) return;
        request_rtf_sum_ += elapsed_seconds * sample_rate_ / double(audio_samples);
        ++completed_requests_;
    }
    void log(const SchedulerStats &, const CacheUsage &, Clock::time_point now);
};
} // namespace tts
