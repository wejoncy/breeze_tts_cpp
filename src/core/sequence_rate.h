#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace tts {
// TextEncoder resamples before the bridge and repeats after it. Its tensor
// extent and WaveNet mask length intentionally follow distinct upstream rules.
struct SequenceRate {
    static constexpr double minimum_speed = .1, maximum_speed = 4.;
    int semantic_repeat, feature_repeat;
    SequenceRate(int semantic, int feature) : semantic_repeat(semantic), feature_repeat(feature) {
        if (semantic < 1 || feature < 1)
            throw std::invalid_argument("invalid acoustic rates");
    }
    static void validate_speed(double speed) {
        if (!std::isfinite(speed) || speed < minimum_speed || speed > maximum_speed)
            throw std::invalid_argument("speed_factor must be between 0.1 and 4");
    }
    static int checked(double frames) {
        if (!std::isfinite(frames) || frames < 0 || frames > INT32_MAX)
            throw std::overflow_error("acoustic frame count");
        return int(frames);
    }
    int hidden_frames(int codes, double speed = 1.) const {
        validate_speed(speed);
        if (codes < 0)
            throw std::invalid_argument("negative semantic code count");
        return checked(!codes        ? 0
                       : speed == 1. ? double(codes) * semantic_repeat
                                     : std::floor(double(codes) * semantic_repeat / speed) + 1);
    }
    int frames(int codes, double speed = 1.) const {
        return checked(double(hidden_frames(codes, speed)) * feature_repeat);
    }
    int valid_frames(int codes, double speed = 1.) const {
        int total = frames(codes, speed);
        return speed == 1. || !codes
                   ? total
                   : checked(std::floor(double(codes) * semantic_repeat * feature_repeat / speed) +
                             1);
    }
    int codes_for_frames(int target, double speed, int limit) const {
        validate_speed(speed);
        if (target < 1 || limit < 1)
            throw std::invalid_argument("invalid acoustic output limit");
        int lower = 1, upper = limit;
        while (lower < upper) {
            int middle = lower + (upper - lower) / 2;
            if (frames(middle, speed) >= target)
                upper = middle;
            else
                lower = middle + 1;
        }
        return lower;
    }
    // max_frames retains its nominal context bound. Generation stops on the
    // first token that reaches it; only that token's overshoot needs storage.
    int capacity(int codes) const {
        return checked(std::min(double(frames(codes, minimum_speed)),
                                double(frames(codes)) + frames(1, minimum_speed)));
    }
};
} // namespace tts
