#pragma once
#include "core/kv_cache_view.h"
#include "core/sequence_layout.h"

namespace tts {
struct AttentionSequence {
    int query_offset, key_offset, query_length, key_length;
};

// Borrowed inputs for one network invocation. The runner builds this on its
// execution thread; layers consume it without owning requests or GPU resources.
// Sequence extents include inactive graph padding. cu_seqlens and attention
// ranges retain the actual boundaries; no caller-supplied packing flag exists.
struct ForwardBatch {
    SequenceLayout sequence;
    const int *positions = nullptr;
    KVCacheView cache;
    const AttentionSequence *bidirectional = nullptr, *causal = nullptr;
    int capacity = 0, prefix = 0;
    int slot = 0, fixed_position = -1;
    bool decode = false;
    int decode_partitions = 1;
    const int *output_rows = nullptr;
    TensorView output;

    ForwardBatch(int batch = 1, int time = 1) : sequence(SequenceLayout::uniform(batch, time)) {}
    explicit ForwardBatch(SequenceLayout layout) : sequence(layout) {}
    int batch() const { return sequence.sequences; }
    int time() const { return sequence.max_length; }

    int rows() const {
        int count = sequence.rows;
        if (batch() < 1 || time() < 1 || count < 1)
            throw std::invalid_argument("invalid forward batch extent");
        return count;
    }
};
} // namespace tts
