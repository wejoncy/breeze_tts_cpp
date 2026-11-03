#pragma once
#include "core/qkv_projection.h"
#include "ops/kernels/attention/cache_address.cuh"
#include "ops/kernels/common/cuda_math.cuh"
namespace tts {
struct QkvProjectionWriteback {
    void *query = nullptr, *keys = nullptr, *values = nullptr, *bias = nullptr;
    int query_stride = 0, query_channels = 0, key_value_channels = 0, capacity = 0;
    CacheLayout cache;
    const int *positions = nullptr;
    QkvProjectionWriteback() = default;
    explicit QkvProjectionWriteback(const QkvProjectionOutput &o)
        : query(o.query.data), keys(o.keys.data), values(o.values.data), bias(o.bias.data),
          query_stride(o.query.ld()), query_channels(o.query.cols),
          key_value_channels(o.key_value_channels), capacity(o.capacity), cache(o.cache),
          positions(o.positions) {}
    template <class T> __device__ void store(T projected, int row, int column) const {
        auto layout = cache;
        T *key = static_cast<T *>(keys), *value = static_cast<T *>(values);
        int position = positions ? positions[row] : 0, lane = row;
        if (layout.sequences) {
            const auto &sequence = layout.sequences[row];
            if (!sequence.position) return;
            position = *sequence.position;
            if (!layout.key_pages) {
                key = reinterpret_cast<T *>(static_cast<char *>(sequence.keys) +
                                            layout.layer * sequence.layer_bytes);
                value = reinterpret_cast<T *>(static_cast<char *>(sequence.values) +
                                              layout.layer * sequence.layer_bytes);
            }
            layout.block_table = sequence.blocks;
            layout.block_stride = 0;
            lane = 0;
        }
        T result = bias ? r<T>(f(projected) + f(static_cast<const T *>(bias)[column]))
                        : projected;
        if (column < query_channels)
            static_cast<T *>(query)[size_t(row) * query_stride + column] = result;
        else if (column < query_channels + key_value_channels)
            cache_pointer<T, 1>(key, lane, position, capacity, key_value_channels, layout,
                                false)[column - query_channels] = result;
        else
            cache_pointer<T, 1>(value, lane, position, capacity, key_value_channels, layout,
                                true)[column - query_channels - key_value_channels] = result;
    }
};
} // namespace tts
