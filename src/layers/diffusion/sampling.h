#pragma once
#include <algorithm>
#include <cmath>
#include <iterator>
#include <stdexcept>

namespace tts {
struct FlowSampling {
    inline static constexpr int step_counts[] = {4, 8, 16, 32};
    int steps;
    float guidance;
    static FlowSampling resolve(int steps, float guidance, FlowSampling defaults) {
        if (!steps)
            steps = defaults.steps;
        if (guidance == -1)
            guidance = defaults.guidance;
        if (std::find(std::begin(step_counts), std::end(step_counts), steps) ==
                std::end(step_counts) ||
            !std::isfinite(guidance) || guidance < 0 || guidance > 2)
            throw std::invalid_argument("V5 requires 4/8/16/32 Euler steps and CFG rate in [0,2]");
        return {steps, guidance};
    }
};
} // namespace tts
