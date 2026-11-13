#pragma once
#include "runtime/scheduled_batch.h"
#include <memory>

namespace tts {
// Engine-thread policy: waiting/running membership and slot/KV reservation.
// No model inputs, PCM, statistics, preparation tasks or transport objects.
class RequestScheduler {
    struct Impl;
    std::unique_ptr<Impl> impl_;
  public:
    RequestScheduler(DeviceContext &, PagedKVCache &, RequestSlots &, int capacity);
    ~RequestScheduler();
    void add(uint64_t request, size_t inputs);
    void ready(uint64_t request, ResourceRequirements);
    void remove(uint64_t request);
    bool contains(uint64_t request) const;
    const ExecutionRow *row(uint64_t id) const;
    size_t available_rows() const;
    std::pair<size_t, size_t> request_counts() const;
    ScheduledBatch schedule();
    void update(uint64_t row, bool done, bool prefilled);
    void clear();
};
} // namespace tts
