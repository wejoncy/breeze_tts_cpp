#pragma once
#include "core/row_capacity.h"
#include <vector>
#include <utility>

namespace tts {
// CPU shape policy only. Bounded padding trades a little row work for replay.
// Zero means eager; no execution path creates a new graph on a cache miss.
class GraphShapes {
    std::vector<int> rows_;
    unsigned padding_denominator_;
    bool fits(int actual, int capacity) const {
        return capacity >= actual && int64_t(capacity - actual) * padding_denominator_ <= actual;
    }

  public:
    explicit GraphShapes(int max_rows, unsigned padding_denominator = 8)
        : padding_denominator_(padding_denominator) {
        if (max_rows < 1 || !padding_denominator)
            throw std::invalid_argument("invalid graph shape policy");
        for (int64_t rows = 1; rows <= max_rows;) {
            int capacity = std::min(max_rows, row_capacity(int(rows)));
            rows_.push_back(capacity);
            rows = int64_t(capacity) + 1;
        }
    }
    const std::vector<int> &rows() const {
        return rows_;
    }
    std::pair<int, int> decoder(int rows, int lanes, int max_branches) const {
        std::pair<int, int> best{};
        for (int capacity : rows_)
            if (fits(rows, capacity))
                for (int branches = 1; branches <= max_branches; ++branches) {
                    int padded_lanes = capacity * branches;
                    if (fits(lanes, padded_lanes) &&
                        (!best.first || padded_lanes < best.second ||
                         (padded_lanes == best.second && capacity < best.first)))
                        best = {capacity, padded_lanes};
                }
        return best;
    }
    int codec(int rows) const {
        for (int capacity : rows_)
            if (fits(rows, capacity))
                return capacity;
        return 0;
    }
};
} // namespace tts
