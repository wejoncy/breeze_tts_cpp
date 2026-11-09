#include "engine/request_scheduler.h"
#include "tts/request.h"
#include "support/profiler.h"
#include <deque>
#include <optional>
#include <unordered_map>

namespace tts {
struct RequestScheduler::Impl {
    struct Request {
        size_t inputs, next = 0;
        std::optional<ResourceRequirements> requirements;
        bool has_next() const { return next < inputs; }
    };
    struct Running { ExecutionRow row; bool prefilled = false; };
    DeviceContext &memory;
    PagedKVCache &kv;
    RequestSlots &slots;
    size_t capacity;
    uint64_t row_sequence = 0;
    std::unordered_map<uint64_t, Request> requests;
    std::deque<uint64_t> waiting;
    std::vector<Running> running;
    std::vector<uint64_t> releases, request_releases;
    Impl(DeviceContext &device, PagedKVCache &cache, RequestSlots &pool, int batch)
        : memory(device), kv(cache), slots(pool), capacity(batch) { running.reserve(capacity); }
    void remove(uint64_t id) {
        if (!requests.erase(id)) return;
        waiting.erase(std::remove(waiting.begin(), waiting.end(), id), waiting.end());
        for (auto it = running.begin(); it != running.end();) {
            if (it->row.request != id) { ++it; continue; }
            releases.push_back(it->row.id);
            it = running.erase(it);
        }
        request_releases.push_back(id);
    }
};
RequestScheduler::RequestScheduler(DeviceContext &memory, PagedKVCache &kv, RequestSlots &slots, int batch)
    : impl_(std::make_unique<Impl>(memory, kv, slots, batch)) {}
RequestScheduler::~RequestScheduler() = default;
void RequestScheduler::add(uint64_t id, size_t inputs) {
    if (!inputs || !impl_->requests.emplace(id, Impl::Request{inputs}).second)
        throw std::logic_error("invalid scheduling request");
    impl_->waiting.push_back(id);
}
void RequestScheduler::ready(uint64_t id, ResourceRequirements requirements) {
    auto &request = impl_->requests.at(id);
    if (request.requirements || requirements.pages.size() != request.inputs)
        throw std::logic_error("invalid scheduling readiness");
    request.requirements = std::move(requirements);
}
void RequestScheduler::remove(uint64_t id) { impl_->remove(id); }
bool RequestScheduler::contains(uint64_t id) const { return impl_->requests.count(id); }
const ExecutionRow *RequestScheduler::row(uint64_t id) const {
    for (const auto &entry : impl_->running) if (entry.row.id == id) return &entry.row;
    return nullptr;
}
size_t RequestScheduler::available_rows() const {
    return std::min(impl_->capacity - impl_->running.size(), impl_->slots.available());
}
ScheduledBatch RequestScheduler::schedule() {
    auto &i = *impl_;
    ScheduledBatch batch;
    // Inspect each waiting request once per free lane. A blocked request must
    // not prevent another ready request from using spare capacity. Engine keeps
    // the selected successor until completion; no per-request retry state is needed.
    size_t examined = 0;
    while (available_rows() && !i.waiting.empty() && examined < i.waiting.size()) {
        uint64_t id = i.waiting.front();
        i.waiting.pop_front();
        auto &request = i.requests.at(id);
        if (!request.requirements) {
            i.waiting.push_back(id); ++examined; continue;
        }
        const size_t input = request.next;
        std::shared_ptr<PagedKVCache::Reservation> cache;
        try {
            const int pages = request.requirements->pages.at(input);
            const size_t workspace = request.requirements->workspace_bytes;
            const size_t retained = i.kv.runtime_storage_bytes();
            const size_t capacity = i.memory.runtime_capacity_bytes();
            const size_t alone_kv = pages > i.kv.capacity()
                ? size_t(pages - i.kv.capacity()) * i.kv.storage_page_bytes() : 0;
            if (pages <= 0 || size_t(pages) > i.kv.max_storage_bytes() / i.kv.storage_page_bytes() ||
                retained > capacity || Tensor::aligned(workspace) > capacity - retained ||
                alone_kv > capacity - retained - Tensor::aligned(workspace))
                throw RequestOverloaded("input exceeds initialized runtime capacity");
            size_t protected_workspace = workspace;
            for (const auto &active : i.running)
                protected_workspace = std::max(protected_workspace,
                    i.requests.at(active.row.request).requirements->workspace_bytes);
            size_t bytes = SIZE_MAX;
            try { bytes = i.kv.additional_bytes(pages); }
            catch (const std::invalid_argument &) {} // Other running rows hold this capacity.
            if (bytes <= i.memory.available_bytes() &&
                Tensor::aligned(protected_workspace) <= i.memory.available_bytes() - bytes &&
                i.memory.can_allocate({bytes, protected_workspace}))
                cache = std::make_shared<PagedKVCache::Reservation>(i.kv, pages);
        } catch (const DeviceError &) { throw; }
        catch (...) {
            batch.rejected.emplace_back(id, std::current_exception());
            i.remove(id);
            examined = 0;
            continue;
        }
        if (!cache) {
            i.waiting.push_back(id); ++examined; continue;
        }
        auto slot = i.slots.acquire();
        ++request.next;
        i.running.push_back({{++i.row_sequence, id, input, true, std::move(cache), std::move(slot)}, false});
        if (input == 0) batch.requests.push_back(id);
        if (request.has_next()) i.waiting.push_back(id);
        examined = 0;
    }
    batch.release.swap(i.releases);
    batch.release_requests.swap(i.request_releases);
    // Eligible waiting rows are admitted above. Execute every pending prefill
    // before decode rows; a busy decoder never spends their prefill budget.
    batch.rows.reserve(i.running.size());
    size_t prefills = 0;
    for (bool prefilled : {false, true}) {
        for (auto &entry : i.running) {
            if (entry.prefilled != prefilled) continue;
            batch.rows.push_back(entry.row);
            if (!prefilled) ++prefills;
            entry.row.create = false;
        }
    }
    mark("tts.scheduler.prefill_rows", prefills);
    mark("tts.scheduler.decode_rows", batch.rows.size() - prefills);
    return batch;
}
void RequestScheduler::update(uint64_t id, bool done, bool prefilled) {
    auto &i = *impl_;
    auto found = std::find_if(i.running.begin(), i.running.end(),
        [&](const auto &entry) { return entry.row.id == id; });
    if (found == i.running.end()) return;
    auto request_id = found->row.request;
    auto &request = i.requests.at(request_id);
    found->prefilled = prefilled;
    if (!done) return;
    i.running.erase(found);
    if (!request.has_next() && std::none_of(i.running.begin(), i.running.end(),
        [&](const auto &entry) { return entry.row.request == request_id; })) i.remove(request_id);
}
void RequestScheduler::clear() {
    impl_->waiting.clear(); impl_->running.clear(); impl_->requests.clear();
    impl_->releases.clear(); impl_->request_releases.clear();
}
} // namespace tts
