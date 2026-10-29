#pragma once
#include "tensor.h"

namespace tts {
// Borrowed metadata only. No allocator, CUDA API, model, engine or graph dependency.
struct CacheLayout {
    const int *block_table = nullptr;
    int block_stride = 0;
    int page_size = 0;
    bool ring = false;
    int page_shift = 0;
    CacheLayout() = default;
    CacheLayout(const int *table, int stride, int tokens, bool circular = false)
        : block_table(table), block_stride(stride), page_size(tokens), ring(circular),
          page_shift(tokens == 64    ? 6
                     : tokens == 128 ? 7
                     : tokens == 256 ? 8
                                     : 0) {}
    // Power-of-two pages compatible with the 64-key FlashAttention tile.
    bool valid_paged() const {
        return block_table && !ring && page_shift >= 6 && page_shift <= 8 &&
               page_size == (1 << page_shift) && block_stride > 0;
    }
};
struct KVCacheView {
    TensorView keys, values;
    int rows_per_layer = 0;
    CacheLayout layout;
};
} // namespace tts
