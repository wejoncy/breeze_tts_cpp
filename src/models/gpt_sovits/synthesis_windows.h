#pragma once
#include "tts/request.h"
#include <cmath>

namespace tts::gpt_sovits {
struct WindowLimits {
    int window, target, core, future, holdback, left, fade;
    double frames_per_second;
};
// CPU-only playback estimate. All observations include semantic waits and CPU
// delivery; no remote playback position is assumed or shared with the engine.
class WindowSchedule {
    int initial_ = 1, maximum_ = 1, last_ = 1;
    double started_ = 0, first_pcm_ = -1, audio_ = 0, cost_ = 0;
  public:
    WindowSchedule() = default;
    WindowSchedule(int initial, int maximum, double now)
        : initial_(initial), maximum_(std::max(initial, maximum)), last_(initial), started_(now) {}
    int choose(int available, double now) const {
        if (first_pcm_ < 0) return std::min(initial_, available);
        double slack = audio_ - (now - first_pcm_);
        int growth = std::min(maximum_, std::max(initial_, int(std::nearbyint(last_ * 1.5))));
        int selected = std::min(initial_, available);
        for (int size = initial_;;) {
            if (size <= std::min(available, growth) && cost_ * size * 1.25 + .1 <= slack)
                selected = size;
            if (size == maximum_) break;
            size = std::min(maximum_, std::max(size + 1, int(std::nearbyint(size * 1.5))));
        }
        return selected;
    }
    void observe(int units, size_t samples, int sample_rate, double now) {
        double rate = (now - started_) / std::max(1, units);
        cost_ = first_pcm_ < 0 ? rate : std::max(rate, cost_ * .8);
        if (first_pcm_ < 0) first_pcm_ = now;
        audio_ += double(samples) / sample_rate;
        last_ = units;
        started_ = now;
    }
};
struct SynthesisWindows {
    static constexpr double maximum_overlap_seconds = .2;
    SynthesisMode mode = SynthesisMode::Fixed;
    int core = 1, future = 0, holdback = 0, fade = 0, left = 0, maximum = 1;
    static SynthesisWindows resolve(const GenerationOptions &o, const WindowLimits &limits,
                                     int reference) {
        SynthesisWindows result;
        result.mode = o.synthesis_mode;
        synthesis_mode_name(result.mode); // Validate enum values from native callers.
        auto frames = [&](double seconds, int fallback, double lower, double upper,
                          double default_marker, const char *name) {
            if (seconds == default_marker) return fallback;
            if (!std::isfinite(seconds) || seconds < lower || seconds > upper)
                throw std::invalid_argument(std::string("invalid ") + name);
            return int(std::nearbyint(seconds * limits.frames_per_second));
        };
        if (limits.window < 1 || limits.target < 1 || reference < 1 ||
            reference >= limits.window || !(limits.frames_per_second > 0) ||
            !std::isfinite(limits.frames_per_second))
            throw std::invalid_argument("invalid synthesis window capacity");
        result.core = frames(o.streaming_chunk_seconds, limits.core, .2, 5., 0,
                             "streaming_chunk_seconds");
        result.future = frames(o.streaming_lookahead_seconds, limits.future, 0, .5, -1,
                               "streaming_lookahead_seconds");
        result.fade = frames(o.streaming_overlap_seconds, limits.fade, 0, maximum_overlap_seconds, -1,
                             "streaming_overlap_seconds");
        result.holdback = frames(o.semantic_lookahead_seconds, limits.holdback, .16, 2., -1,
                                 "semantic_lookahead_seconds");
        result.maximum = std::min(limits.window - reference, limits.target);
        result.left = std::max(limits.left, result.fade);
        if (result.mode == SynthesisMode::Offline) {
            result.core = result.maximum;
            result.future = result.fade = result.holdback = result.left = 0;
        } else {
            if (result.core < 1 || result.future < 0 || result.holdback < 0 || result.fade < 0 ||
                result.left < 0 || result.core + result.future > result.maximum)
                throw std::invalid_argument("reference plus streaming window exceeds model context");
            result.fade = std::min(result.fade, result.core - 1);
        }
        return result;
    }
    int lookahead(int cursor) const {
        return mode == SynthesisMode::Hybrid && cursor ? 0 : future;
    }
    int ready(int total, int cursor, bool finished, const WindowSchedule &schedule,
              double now) const {
        if (mode == SynthesisMode::Offline && !finished) return 0;
        int safe = finished ? total : total - lookahead(cursor) - holdback;
        if (safe <= cursor || (!finished && safe - cursor < core)) return 0;
        int units = mode == SynthesisMode::Hybrid
                        ? schedule.choose(safe - cursor, now) : core;
        if (!finished && safe - cursor < units) return 0;
        return std::min(units, safe - cursor);
    }
};
} // namespace tts::gpt_sovits
