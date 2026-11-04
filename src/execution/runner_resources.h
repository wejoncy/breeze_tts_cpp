#pragma once
#include "runtime/execution_resources.h"
#include "runtime/graph_shapes.h"
#include "runtime/request_admission.h"
#include "runtime/request_slots.h"
#include "runtime/state_pool.h"

namespace tts {
// One owner for execution storage and request capacity. Network code borrows
// tensor views; it neither creates pools nor owns their lifetime.
struct RunnerResources {
    struct Cache {
        int layers, channels, sequences_per_input;
        Type type;
    };
    DeviceContext &memory;
    const int batch, context;
    PagedKVCache kv;
    RequestAdmissionPool admissions;
    RequestSlots slots;
    StatePool states;
    GraphShapes shapes;
    ExecutionResources execution;
    RunnerResources(DeviceContext &device, Cache cache, int batch_capacity, int context_capacity,
                    int page_tokens)
        : memory(device), batch(batch_capacity), context(context_capacity),
          kv(device, cache.layers, cache.channels, cache.type,
             batch_capacity * cache.sequences_per_input, context_capacity, page_tokens),
          admissions(device, kv, batch_capacity), slots(batch_capacity), states(device), shapes(batch_capacity),
          execution(device) {}
};
} // namespace tts
