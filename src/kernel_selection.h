#pragma once
#include "kernel_problem.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>

namespace tts {
// CPU-only policy. The injected search owns execution, validation and timing.
// No tensor, CUDA, allocator, model, graph, or serving dependency is allowed here.
// Cached choices are values; callers never borrow an entry's lifetime.
template <class Choice, class Operation> class KernelSelector {
    using Key = std::tuple<Operation, int, int, int>;
    struct Entry {
        Choice choice;
        uint64_t use;
    };
    std::map<Key, Entry> entries_;
    size_t capacity_;
    unsigned bins_per_octave_;
    uint64_t clock_ = 0;
    SelectionStats stats_;

  public:
    explicit KernelSelector(size_t capacity = 64, unsigned bins_per_octave = 8)
        : capacity_(capacity), bins_per_octave_(bins_per_octave) {
        if (!capacity || !bins_per_octave)
            throw std::invalid_argument("kernel selection limits must be positive");
    }
    static KernelProblem representative(KernelProblem p, unsigned bins_per_octave = 8) {
        if (p.rows < 1 || p.channels < 1 || p.reduction < 1)
            throw std::invalid_argument("invalid kernel problem");
        // Ceil to a capacity class, not an execution shape or a padding request.
        // The upper bound also makes workspace validation safe for every member.
        if (!bins_per_octave)
            throw std::invalid_argument("invalid selection class resolution");
        int64_t upper = 1;
        while (upper < p.rows)
            upper *= 2;
        int64_t step = std::max<int64_t>(1, upper / (2ull * bins_per_octave));
        int64_t rows = (int64_t(p.rows) + step - 1) / step * step;
        p.rows = int(std::min<int64_t>(rows, std::numeric_limits<int>::max()));
        return p;
    }
    template <class Search>
    Choice select(Operation operation, KernelProblem actual, Search &&search) {
        auto p = representative(actual, bins_per_octave_);
        Key key{operation, p.rows, p.channels, p.reduction};
        auto it = entries_.find(key);
        if (it != entries_.end()) {
            it->second.use = ++clock_;
            ++stats_.hits;
            return it->second.choice;
        }
        // Exceptions do not publish partial/invalid selections.
        Choice choice = search(p);
        ++stats_.searches;
        if (entries_.size() == capacity_) {
            auto oldest = std::min_element(
                entries_.begin(), entries_.end(),
                [](const auto &a, const auto &b) { return a.second.use < b.second.use; });
            entries_.erase(oldest);
            ++stats_.evictions;
        }
        entries_.emplace(key, Entry{choice, ++clock_});
        return choice;
    }
    SelectionStats stats() const {
        auto result = stats_;
        result.entries = entries_.size();
        return result;
    }
};
} // namespace tts
