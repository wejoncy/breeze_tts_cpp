#pragma once
#include <cstddef>

namespace tts {
class DeviceContext;
// Init-time CUDA queries only; request logs use cached device capacity and KV counters.
size_t log_device_memory(const DeviceContext &, const char *phase);
} // namespace tts
