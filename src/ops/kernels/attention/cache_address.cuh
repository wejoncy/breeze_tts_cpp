#pragma once
#include "core/kv_cache_view.h"

namespace tts {
template <int Mode>
__device__ size_t cache_row(int lane, int position, int capacity, CacheLayout cache) {
    if constexpr (Mode == 1)
        return size_t(cache.block_table[lane * cache.block_stride +
                                       (position >> cache.page_shift)]) * cache.page_size +
               (position & (cache.page_size - 1));
    if constexpr (Mode == 2)
        return size_t(lane) * capacity + position % capacity;
    return size_t(lane) * capacity + position;
}
template <class T, int Mode>
__device__ T *cache_pointer(T *base, int lane, int position, int capacity, int width,
                            CacheLayout cache, bool value) {
    if constexpr (Mode == 1) {
        if (cache.key_pages) {
            int page = cache.block_table[lane * cache.block_stride +
                                         (position >> cache.page_shift)];
            auto *address = (value ? cache.value_pages : cache.key_pages)[page];
            return reinterpret_cast<T *>(static_cast<char *>(address) +
                                         cache.layer * cache.page_layer_bytes) +
                   (position & (cache.page_size - 1)) * width;
        }
    }
    return base + cache_row<Mode>(lane, position, capacity, cache) * width;
}
} // namespace tts
