#pragma once
#include "runtime/kv_cache.h"
#include "runtime/request_slots.h"
#include <exception>

namespace tts {
struct ResourceRequirements {
    std::vector<int> pages;
    size_t workspace_bytes = 0;
};
struct ExecutionRow {
    uint64_t id;
    uint64_t request;
    size_t input;
    bool create = true;
    std::shared_ptr<PagedKVCache::Reservation> cache;
    std::shared_ptr<RequestSlots::Lease> slot;
};
// Engine attaches model inputs and completion storage to these reserved rows.
struct ScheduledBatch {
    std::vector<uint64_t> requests;
    std::vector<ExecutionRow> rows;
    std::vector<uint64_t> release, release_requests;
    std::vector<std::pair<uint64_t, std::exception_ptr>> rejected;
};
} // namespace tts
