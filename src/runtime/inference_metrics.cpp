#include "runtime/inference_metrics.h"
#include "support/log.h"
#include "tts/request.h"

namespace tts {
void InferenceMetrics::record_output(uint64_t request, size_t samples, const GenerationFrame &frame) {
    if (samples) request_audio_samples_[request] += samples;
    if (frame.done) {
        if (!frame.cancelled && !frame.error) record_rtf(request, frame.stats.total_ms);
        request_audio_samples_.erase(request);
    }
}

void InferenceMetrics::record_rtf(uint64_t request, double elapsed_ms) {
    const auto it = request_audio_samples_.find(request);
    if (it == request_audio_samples_.end()) return;
    request_rtf_sum_ += elapsed_ms / 1000 * sample_rate_ / double(it->second);
    ++measured_requests_;
}

void InferenceMetrics::log(const SchedulerStats &scheduler, const CacheUsage &cache,
                           Clock::time_point now) {
    const double seconds = std::chrono::duration<double>(now - last_log_).count();
    const bool activity = audio_samples_ || measured_requests_ ||
                          scheduler.running_requests || scheduler.waiting_requests;
    const auto level = activity || last_activity_ ? TTS_LOG_INFO : TTS_LOG_DEBUG;
    const size_t used = size_t(cache.used_pages) * cache.page_bytes + cache.active_rows * cache.state_bytes;
    const size_t backed = size_t(cache.backed_pages) * cache.page_bytes + cache.state_rows * cache.state_bytes;
    peak_bytes_ = std::max(peak_bytes_, used);
    constexpr double mib = 1024.0 * 1024.0;
    const auto percent = [](size_t used, size_t total) {
        return total ? 100.0 * double(used) / double(total) : 0.0;
    };
    TTS_LOG(level, "engine.metrics",
            "Avg audio throughput: {:.2f} audio s/s{}, "
            "Running: {} reqs, Waiting: {} reqs, GPU KV cache usage: {:.1f}%, "
            "used={:.0f} MiB observed_peak={:.0f} MiB device_used={:.2f}%",
            double(audio_samples_) / sample_rate_ / seconds,
            measured_requests_ ? fmt::format(", Avg RTF: {:.4f}", request_rtf_sum_ / measured_requests_) : "",
            scheduler.running_requests, scheduler.waiting_requests, percent(used, backed),
            used / mib, peak_bytes_ / mib, percent(used, cache.device_bytes));
    audio_samples_ = 0;
    request_rtf_sum_ = 0;
    measured_requests_ = 0;
    last_active_rows_ = cache.active_rows;
    last_activity_ = activity;
    last_log_ = now;
}
} // namespace tts
