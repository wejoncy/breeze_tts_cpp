#pragma once
#include "tts/request.h"
#include <numeric>

namespace tts {
// Executor-owned fixed slot IDs. A lease holds stable init-time state storage;
// moving a request in a batch only changes metadata, never its slot or contents.
class RequestSlots {
    std::vector<size_t> free_;

  public:
    explicit RequestSlots(size_t count) : free_(count) {
        std::iota(free_.rbegin(), free_.rend(), size_t(0));
    }
    struct Lease {
        RequestSlots &pool;
        const size_t index;
        ~Lease() { pool.free_.push_back(index); }
        Lease(const Lease &) = delete;

      private:
        friend class RequestSlots;
        Lease(RequestSlots &owner, size_t i) : pool(owner), index(i) {}
    };
    std::unique_ptr<Lease> acquire() {
        if (free_.empty())
            throw RequestOverloaded("request state capacity exhausted");
        auto lease = std::unique_ptr<Lease>(new Lease(*this, free_.back()));
        free_.pop_back();
        return lease;
    }
};
} // namespace tts
