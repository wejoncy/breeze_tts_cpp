#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace tts {
inline size_t aligned_storage_bytes(size_t bytes) {
    if (bytes > SIZE_MAX - 255) throw std::overflow_error("tensor allocation size");
    return (bytes + 255) & ~size_t(255);
}
} // namespace tts
