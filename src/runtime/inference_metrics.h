#pragma once
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <unordered_map>

namespace tts {
struct GenerationFrame;
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
// Engine-thread metrics. Request bookkeeping and aggregation stay in this collector.
class InferenceMetrics {
  public:
    using Clock = std::chrono::steady_clock;

  private:
    Clock::time_point last_log_;
    const int sample_rate_;
    uint64_t audio_samples_ = 0;
    std::unordered_map<uint64_t, uint64_t> request_audio_samples_;
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
    void record_output(uint64_t request, const GenerationFrame &);
    void retire(uint64_t request) { request_audio_samples_.erase(request); }
    void log(const SchedulerStats &, const CacheUsage &, Clock::time_point now);
};
} // namespace tts
