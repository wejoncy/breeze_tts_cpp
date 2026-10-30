#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace tts {
// Logarithmic capacity classes shared by resource planning and kernel policy.
// This computes a bound only; callers decide whether execution needs padding.
inline int row_capacity(int rows, unsigned bins_per_octave = 8) {
    if (rows < 1 || !bins_per_octave)
        throw std::invalid_argument("invalid row capacity class");
    int64_t upper = 1;
    while (upper < rows)
        upper *= 2;
    int64_t step = std::max<int64_t>(1, upper / (2ull * bins_per_octave));
    int64_t capacity = (int64_t(rows) + step - 1) / step * step;
    return int(std::min<int64_t>(capacity, std::numeric_limits<int>::max()));
}
} // namespace tts
