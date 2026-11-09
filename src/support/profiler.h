#pragma once
#include <nvtx3/nvToolsExt.h>
#include <cstdint>
#include <string_view>

namespace tts {
// A frame tag is diagnostic metadata only; no routing or lifecycle uses it.
inline uint64_t frame_trace_id(std::string_view request, uint32_t frame) {
    uint32_t hash = 2166136261u;
    for (unsigned char byte : request) hash = (hash ^ byte) * 16777619u;
    return (uint64_t(hash) << 32) | frame;
}
inline void mark(const char *name, uint64_t value) {
    nvtxEventAttributes_t event{};
    event.version = NVTX_VERSION;
    event.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    event.messageType = NVTX_MESSAGE_TYPE_ASCII;
    event.message.ascii = name;
    event.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
    event.payload.ullValue = value;
    nvtxMarkEx(&event);
}
struct Range {
    explicit Range(const char *name) { nvtxRangePushA(name); }
    ~Range() { nvtxRangePop(); }
    Range(const Range &) = delete;
    Range &operator=(const Range &) = delete;
};
} // namespace tts
