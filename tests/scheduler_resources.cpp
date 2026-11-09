#include "engine/request_scheduler.h"
#include "core/tensor_storage.h"
#include <iostream>
#include <set>

static void require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}
int main() {
    try {
        tts::DeviceContext memory(0, size_t(64) << 20);
        tts::Tensor::Scope allocation(memory);
        tts::PagedKVCache kv(memory, 1, 32, tts::Type::F32, 4, 256, 64);
        memory.reserve_transfers(size_t(1) << 20);
        memory.reserve_runtime(size_t(1) << 20);
        tts::RequestSlots slots(4);
        tts::RequestScheduler scheduler(memory, kv, slots, 4);
        // An unprepared older request cannot block a ready request. Reserve
        // enough workspace that only two of this request's nine rows fit.
        scheduler.add(1, 1);
        scheduler.add(2, 9);
        const size_t workspace = memory.runtime_capacity_bytes() - 8 * kv.storage_page_bytes();
        scheduler.ready(2, {std::vector<int>(9, 4), workspace});
        auto batch = scheduler.schedule();
        require(batch.rows.size() == 2 && batch.requests.size() == 1,
                "KV reservation did not constrain the running batch");
        require(kv.committed() == 8 && kv.capacity() >= 8,
                "scheduler returned rows before physically backing their KV capacity");
        // Each running row reserves its cache. Waiting membership
        // remains in Scheduler and every input is selected exactly once.
        std::set<size_t> completed;
        while (!batch.rows.empty()) {
            for (const auto &row : batch.rows) {
                require(row.request == 2 && completed.insert(row.input).second,
                        "resource-constrained input wave was lost or repeated");
                scheduler.update(row.id, true, true);
            }
            batch.rows.clear(); // The completed batch no longer owns cache capacity.
            batch = scheduler.schedule();
            require(batch.rows.size() <= 2, "request exceeded the memory budget");
        }
        require(completed.size() == 9 && !scheduler.contains(2) && scheduler.contains(1),
                "scheduler retired the wrong waiting/running request");
        scheduler.ready(1, {{1}, 0});
        batch = scheduler.schedule();
        require(batch.rows.size() == 1 && batch.rows.front().request == 1,
                "released capacity did not admit the waiting request");
        auto live_cache = batch.rows.front().cache;
        auto live_slot = batch.rows.front().slot;
        scheduler.remove(1);
        batch.rows.clear();
        require(live_cache && kv.committed() == 1, "cancel invalidated in-flight cache capacity");
        live_cache.reset();
        require(kv.committed() == 0, "completed cache capacity was not released");
        require(slots.available() == 3, "cancel recycled an in-flight slot");
        live_slot.reset();
        require(slots.available() == 4, "completed slot was not released");
        std::vector<std::shared_ptr<tts::RequestSlots::Lease>> held_slots;
        while (slots.available()) held_slots.push_back(slots.acquire());
        scheduler.add(5, 1);
        scheduler.ready(5, {{1}, 0});
        require(scheduler.available_rows() == 0 && scheduler.schedule().rows.empty(),
                "scheduler admitted a row without a free slot");
        held_slots.pop_back();
        batch = scheduler.schedule();
        require(batch.rows.size() == 1 && batch.rows.front().slot && slots.available() == 0,
                "scheduler did not reserve the released slot before returning its row");
        scheduler.remove(5);
        batch.rows.clear();
        held_slots.clear();
        scheduler.add(3, 1);
        scheduler.ready(3, {{100}, 0});
        scheduler.add(4, 1);
        scheduler.ready(4, {{1}, 0});
        batch = scheduler.schedule();
        require(batch.rejected.size() == 1 && batch.rejected.front().first == 3 &&
                batch.rows.size() == 1 && batch.rows.front().request == 4,
                "impossible request corrupted a schedulable neighbor");
        scheduler.update(batch.rows.front().id, false, true);
        scheduler.add(6, 1);
        scheduler.ready(6, {{1}, 0});
        batch = scheduler.schedule();
        require(batch.rows.size() == 2 && batch.rows.front().request == 6 && batch.rows.back().request == 4,
                "eligible prefill did not take priority over a running decoder");
        memory.sync();
        std::cout << "scheduler KV reservation, input waves and lease lifetime checks passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
