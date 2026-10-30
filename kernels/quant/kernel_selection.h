#pragma once
#include "kernels/quant/kernel_problem.h"
#include "core/row_capacity.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>

namespace tts {
// CPU-only policy. The injected policy selects a configuration for each row class.
// No tensor, CUDA, allocator, model, graph, or serving dependency is allowed here.
// Choices live for the selector lifetime. Row classes grow logarithmically;
// Choices remain available to every prepared plan in the class.
// GPU plans/descriptors have their own bounded cache and do not own these choices.
template <class Choice, class Operation> class KernelSelector {
    using Key = std::tuple<Operation, int, int, int>;
    std::map<Key, Choice> entries_;
    unsigned bins_per_octave_;
    SelectionStats stats_;

  public:
    explicit KernelSelector(unsigned bins_per_octave = 8) : bins_per_octave_(bins_per_octave) {
        if (!bins_per_octave)
            throw std::invalid_argument("selection class resolution must be positive");
    }
    static KernelProblem representative(KernelProblem p, unsigned bins_per_octave = 8) {
        if (p.rows < 1 || p.channels < 1 || p.reduction < 1)
            throw std::invalid_argument("invalid kernel problem");
        p.rows = row_capacity(p.rows, bins_per_octave);
        return p;
    }
    template <class Search>
    Choice select(Operation operation, KernelProblem actual, Search &&search) {
        auto p = representative(actual, bins_per_octave_);
        Key key{operation, p.rows, p.channels, p.reduction};
        auto it = entries_.find(key);
        if (it != entries_.end()) {
            ++stats_.hits;
            return it->second;
        }
        // Exceptions do not publish partial/invalid selections.
        Choice choice = search(p);
        entries_.emplace(key, choice);
        ++stats_.searches;
        return choice;
    }
    SelectionStats stats() const {
        auto result = stats_;
        result.entries = entries_.size();
        return result;
    }
};
} // namespace tts
