#pragma once
#include "core/sequence_layout.h"
#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

namespace tts {
// CPU sequence descriptors. reserve() owns the allocation; append() and padding
// only fill that backing. Exact-size packing uses the same layout without copies.
class SequenceBuilder {
    std::vector<int> data_;
    int row_capacity_ = 0, sequence_capacity_ = 0;
    int rows_ = 0, sequences_ = 0, maximum_ = 0;
    int *batches() { return data_.data(); }
    int *positions() { return data_.data() + row_capacity_; }
    int *offsets() { return data_.data() + size_t(row_capacity_) * 2; }
    int *lengths() { return offsets() + sequence_capacity_ + 1; }

  public:
    void reserve(int sequences, int rows) {
        if (rows < 0 || sequences < 0 || rows_ || sequences_)
            throw std::invalid_argument("invalid sequence builder reservation");
        if (rows <= row_capacity_ && sequences <= sequence_capacity_) return;
        row_capacity_ = std::max(rows, row_capacity_);
        sequence_capacity_ = std::max(sequences, sequence_capacity_);
        data_.resize(2 * size_t(row_capacity_) + 2 * size_t(sequence_capacity_) + 1);
        offsets()[0] = 0;
    }
    void clear() {
        rows_ = sequences_ = maximum_ = 0;
        if (!data_.empty()) offsets()[0] = 0;
    }
    void append(int extent, int valid) {
        if (extent < 0 || valid < 0 || valid > extent ||
            extent > row_capacity_ - rows_ || sequences_ == sequence_capacity_)
            throw std::invalid_argument("sequence exceeds reserved metadata capacity");
        std::fill_n(batches() + rows_, extent, sequences_);
        std::iota(positions() + rows_, positions() + rows_ + extent, 0);
        rows_ += extent;
        lengths()[sequences_] = valid;
        offsets()[++sequences_] = rows_;
        maximum_ = std::max(maximum_, extent);
    }
    int rows() const { return rows_; }
    int size() const { return sequences_; }
    SequenceLayout view() const {
        if (data_.empty()) return {};
        const auto *p = data_.data();
        return {p, p + row_capacity_, p + size_t(row_capacity_) * 2,
                p + size_t(row_capacity_) * 2 + sequence_capacity_ + 1,
                rows_, sequences_, maximum_};
    }
    SequenceLayout padded_view(int rows, int sequences) {
        if (rows < rows_ || rows > row_capacity_ || sequences < sequences_ ||
            sequences > sequence_capacity_)
            throw std::invalid_argument("invalid sequence metadata padding");
        if (data_.empty()) return {};
        std::fill(batches() + rows_, batches() + rows, -1);
        std::fill(positions() + rows_, positions() + rows, 0);
        std::fill(offsets() + sequences_ + 1, offsets() + sequences + 1, rows_);
        std::fill(lengths() + sequences_, lengths() + sequences, 0);
        auto out = view();
        out.rows = rows;
        out.sequences = sequences;
        return out;
    }
    bool uniform() const {
        const auto out = view();
        return maximum_ > 0 && int64_t(maximum_) * sequences_ == rows_ &&
               std::all_of(out.lengths, out.lengths + sequences_,
                           [&](int n) { return n == maximum_; });
    }
    bool same_extents(const SequenceBuilder &other) const {
        if (!sequences_ || !other.sequences_) return sequences_ == other.sequences_;
        return sequences_ == other.sequences_ && rows_ == other.rows_ &&
               std::equal(view().cu_seqlens, view().cu_seqlens + sequences_ + 1,
                          other.view().cu_seqlens);
    }
    std::vector<int> release() && {
        if (rows_ != row_capacity_ || sequences_ != sequence_capacity_)
            throw std::logic_error("only exact-size sequence metadata can be released");
        if (data_.empty()) return {0};
        return std::move(data_);
    }
    static std::vector<int> pack(const std::vector<int> &lengths) {
        int64_t rows = 0;
        for (int n : lengths) {
            if (n < 0 || (rows += n) > INT32_MAX)
                throw std::invalid_argument("invalid packed sequence lengths");
        }
        if (lengths.size() > size_t(INT32_MAX))
            throw std::invalid_argument("too many sequences");
        SequenceBuilder builder;
        builder.reserve(int(lengths.size()), int(rows));
        for (int n : lengths) builder.append(n, n);
        return std::move(builder).release();
    }
};
} // namespace tts
