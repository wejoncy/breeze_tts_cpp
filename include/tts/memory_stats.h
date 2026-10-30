#pragma once
#include <cstdint>
namespace tts {
// Engine-owned CUDA pool only; driver and library-private allocations are excluded.
struct MemoryStats {
    uint64_t used_bytes = 0, reserved_bytes = 0, peak_used_bytes = 0, peak_reserved_bytes = 0;
};
} // namespace tts
