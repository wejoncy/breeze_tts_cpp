#pragma once
#include <cstddef>
#include <vector>

namespace tts {
struct SequenceStateSpec {
    size_t kv_bytes = 0;
    size_t layer_bytes = 0;
    std::vector<size_t> history_bytes;
};
} // namespace tts
