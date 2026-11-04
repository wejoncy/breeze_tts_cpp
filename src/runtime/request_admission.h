#pragma once
#include "runtime/kv_cache.h"
#include "tts/request.h"
#include <set>

namespace tts {
// Capacity credits protect future input waves as well as currently active rows.
// A model supplies workspace bytes and per-input KV page requirements; this pool
// owns the accounting and physical admission policy independently of that model.
class RequestAdmissionPool {
    DeviceContext &memory_;
    PagedKVCache &kv_;
    const size_t capacity_;
    std::multiset<size_t> workspace_;
    std::multiset<int> pages_;

  public:
    RequestAdmissionPool(DeviceContext &memory, PagedKVCache &kv, size_t capacity)
        : memory_(memory), kv_(kv), capacity_(capacity) {}
    class Lease {
        RequestAdmissionPool &pool_;
        std::multiset<size_t>::iterator workspace_;
        std::vector<std::multiset<int>::iterator> pages_;
        size_t active_rows_ = 0;
        friend class RequestAdmissionPool;
        Lease(RequestAdmissionPool &pool, size_t workspace, const std::vector<int> &pages,
              int peak_pages) : pool_(pool) {
            workspace_ = pool_.workspace_.insert(workspace);
            try {
                pages_.reserve(pages.size());
                for (int count : pages)
                    pages_.push_back(pool_.pages_.insert(count));
                pool_.kv_.reserve(peak_pages);
            } catch (...) {
                release();
                throw;
            }
        }
        void release() noexcept {
            if (active_rows_) std::terminate();
            pool_.workspace_.erase(workspace_);
            for (auto it : pages_)
                pool_.pages_.erase(it);
        }

      public:
        bool belongs_to(const RequestAdmissionPool &pool) const { return &pool_ == &pool; }
        // Executor-owned row credits. A state's KV/slot leases must be destroyed
        // before its row credit, and all states must die before this admission.
        class RowLease {
            Lease &owner_;
            friend class Lease;
            explicit RowLease(Lease &owner) : owner_(owner) {}
          public:
            ~RowLease() { --owner_.active_rows_; }
            RowLease(const RowLease &) = delete;
        };
        std::unique_ptr<RowLease> acquire_row() {
            if (active_rows_ == pages_.size()) return {};
            auto row = std::unique_ptr<RowLease>(new RowLease(*this));
            ++active_rows_;
            return row;
        }
        ~Lease() { release(); }
        Lease(const Lease &) = delete;
    };
    std::unique_ptr<Lease> admit(const std::vector<int> &pages, size_t workspace) {
        if (pages.empty() || !capacity_ ||
            std::any_of(pages.begin(), pages.end(), [](int count) { return count <= 0; }))
            throw RequestOverloaded("invalid KV capacity credits");
        std::vector<int> credits = pages;
        std::sort(credits.begin(), credits.end(), std::greater<int>());
        credits.resize(std::min(credits.size(), capacity_));
        const size_t retained = kv_.runtime_storage_bytes();
        const size_t capacity = memory_.runtime_capacity_bytes();
        const size_t alone_workspace = Tensor::aligned(workspace);
        const size_t alone_kv = kv_.additional_bytes(std::max(0, credits.front() - kv_.committed()));
        // Only permanent KV backing and a single largest input determine
        // impossibility. Current prefill workspace pressure can only defer.
        if (retained > capacity || alone_workspace > capacity - retained ||
            alone_kv > capacity - retained - alone_workspace)
            throw RequestOverloaded("request exceeds initialized runtime memory capacity");

        // Largest global capacity_ credits cover every allowed active subset.
        // For k new inputs, choose j of its descending credits and at most
        // capacity_-j existing credits. Prefix maxima compute every k in O(B).
        std::vector<size_t> existing(capacity_ + 1), peaks(credits.size() + 1);
        auto previous = pages_.rbegin();
        for (size_t i = 1; i <= capacity_; ++i) {
            existing[i] = existing[i - 1];
            if (previous != pages_.rend()) existing[i] += size_t(*previous++);
        }
        peaks[0] = existing.back();
        size_t added = 0;
        for (size_t k = 1; k <= credits.size(); ++k) {
            added += size_t(credits[k - 1]);
            peaks[k] = std::max(peaks[k - 1], added + existing[capacity_ - k]);
        }
        const size_t protected_workspace = workspace_.empty()
            ? workspace : std::max(workspace, *workspace_.rbegin());
        const size_t aligned = Tensor::aligned(protected_workspace);
        const size_t available = memory_.available_bytes();
        for (size_t k = credits.size(); k; --k) {
            if (peaks[k] > size_t(INT_MAX)) continue;
            const int peak = int(peaks[k]);
            size_t kv_bytes;
            try {
                kv_bytes = kv_.additional_bytes(std::max(0, peak - kv_.committed()));
            } catch (const std::invalid_argument &) {
                continue;
            }
            if (aligned > available || kv_bytes > available - aligned ||
                !memory_.can_allocate({kv_bytes, protected_workspace}))
                continue;
            credits.resize(k);
            return std::unique_ptr<Lease>(new Lease(*this, workspace, credits, peak));
        }
        if (!workspace_.empty()) return {};
        throw RequestOverloaded("insufficient initialized runtime memory for one input");
    }
};
} // namespace tts
