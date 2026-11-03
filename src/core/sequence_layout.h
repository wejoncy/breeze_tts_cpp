#pragma once
#include <cstdint>
#include <stdexcept>

namespace tts {
// Device prefix sums describe the physical sequence boundaries. lengths may
// mask a tail inside an extent; row maps are optional addressing accelerators.
// Uniform sequences derive their boundaries from shape and need no arrays.
struct SequenceLayout {
    const int *row_batch = nullptr, *row_position = nullptr;
    const int *cu_seqlens = nullptr, *lengths = nullptr;
    int rows = 0, sequences = 0, max_length = 0;

    static SequenceLayout uniform(int batch, int time) {
        if (batch < 1 || time < 1 || int64_t(batch) * time > INT32_MAX)
            throw std::invalid_argument("invalid sequence extent");
        return {nullptr, nullptr, nullptr, nullptr, batch * time, batch, time};
    }
    const SequenceLayout *addressing() const {
        return cu_seqlens || row_batch ? this : nullptr;
    }
};
} // namespace tts
