#pragma once
#include "core/tensor.h"

namespace tts {
// Borrowed metadata only. No allocator, CUDA API, model, engine or graph dependency.
struct CacheSequence {
    void *keys = nullptr, *values = nullptr;
    size_t layer_bytes = 0;
    const int *blocks = nullptr;
    int *position = nullptr; // null marks an inactive graph-padding lane
};
struct CacheLayout {
    // Dense compute rows borrow persistent sequence storage; kernels resolve it directly.
    const CacheSequence *sequences = nullptr;
    int layer = 0;
    void *const *key_pages = nullptr, *const *value_pages = nullptr;
    size_t page_layer_bytes = 0;

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
        return (block_table || sequences) && !ring && page_shift >= 6 && page_shift <= 8 &&
               page_size == (1 << page_shift) && block_stride > 0;
    }
};
struct KVCacheView {
    TensorView keys, values;
    int rows_per_layer = 0;
    CacheLayout layout;
};
} // namespace tts
