#pragma once
#include <nvtx3/nvToolsExt.h>

namespace tts {
struct Range {
    explicit Range(const char *name) { nvtxRangePushA(name); }
    ~Range() { nvtxRangePop(); }
    Range(const Range &) = delete;
    Range &operator=(const Range &) = delete;
};
} // namespace tts
