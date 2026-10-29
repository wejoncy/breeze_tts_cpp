#pragma once
#include <cstddef>
namespace tts {
struct KernelProblem {
    int rows, channels, reduction;
};
struct SelectionStats {
    size_t entries = 0, hits = 0, searches = 0, evictions = 0;
};
} // namespace tts
