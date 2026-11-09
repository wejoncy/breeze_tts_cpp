#pragma once
#include "core/forward_context.h"
#include <vector>

namespace tts {
// Borrowed device results and event-completed host results. Sampling has no
// dependency on request planning, scheduling policy, or model resources.
struct ForwardOutputs { std::vector<NamedTensor> values; };
struct SampledOutputs { std::vector<NamedTensor> values; };
} // namespace tts
