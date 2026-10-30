#pragma once
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>

namespace tts {
// Exact-shape resources have a different lifetime from kernel policy choices.
// All plans admitted for the current request are pinned; only idle plans may be
// evicted. The bound is current-request plans + idle_capacity, not total batches
// ever seen. Cache management occurs at admission, never during execution.
template <class Key, class Value> class PreparedCache {
    struct Entry {
        std::unique_ptr<Value> value;
        uint64_t epoch, use;
    };
    std::map<Key, Entry> entries_;
    size_t idle_capacity_;
    uint64_t epoch_ = 1, clock_ = 0;
    uint64_t evictions_ = 0;
    bool sealed_ = false;
    void trim() {
        if (sealed_)
            return;
        for (;;) {
            size_t idle = 0;
            auto oldest = entries_.end();
            for (auto it = entries_.begin(); it != entries_.end(); ++it) {
                if (it->second.epoch == epoch_)
                    continue;
                ++idle;
                if (oldest == entries_.end() || it->second.use < oldest->second.use)
                    oldest = it;
            }
            if (idle <= idle_capacity_)
                return;
            entries_.erase(oldest);
            ++evictions_;
        }
    }

  public:
    explicit PreparedCache(size_t idle_capacity = 64) : idle_capacity_(idle_capacity) {}
    void begin_admission() {
        if (sealed_)
            return;
        ++epoch_;
        trim();
    }
    // Initialization publishes the complete working set. Serving only looks up
    // retained plans; a missing shape cannot silently tune or create resources.
    void seal() {
        sealed_ = true;
    }
    template <class Create> Value &prepare(const Key &key, Create &&create) {
        auto it = entries_.find(key);
        if (it == entries_.end()) {
            if (sealed_)
                throw std::logic_error("prepared resource missing after initialization");
            auto value = create();
            if (!value)
                throw std::logic_error("null prepared plan");
            it = entries_.emplace(key, Entry{std::move(value), epoch_, ++clock_}).first;
        } else {
            it->second.epoch = epoch_;
            it->second.use = ++clock_;
        }
        trim();
        return *it->second.value;
    }
    Value *find(const Key &key) const {
        auto it = entries_.find(key);
        return it == entries_.end() || (!sealed_ && it->second.epoch != epoch_)
                   ? nullptr
                   : it->second.value.get();
    }
    size_t size() const {
        return entries_.size();
    }
    uint64_t evictions() const {
        return evictions_;
    }
};
} // namespace tts
