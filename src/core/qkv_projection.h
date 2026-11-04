#pragma once
#include "core/kv_cache_view.h"
namespace tts {
// Borrowed projection destinations. KV channels remain explicit when page
// pointers supply storage and the dense key/value views are empty.
struct QkvProjectionOutput {
    TensorView query, keys, values, bias;
    CacheLayout cache;
    const int *positions = nullptr;
    int key_value_channels = 0, capacity = 0;
    void validate(TensorView input, TensorView weight) const {
        int64_t columns = int64_t(query.cols) + 2 * int64_t(key_value_channels);
        if (!input.data || !weight.data || input.rows < 1 || input.cols < 1 ||
            query.cols < 1 || !query.data || query.rows != input.rows ||
            query.type != input.type ||
            (weight.type != input.type && weight.type != Type::I8 && weight.type != Type::FP8) ||
            (input.type != Type::F16 && input.type != Type::BF16) ||
            input.cols != weight.cols || !weight.dense() || key_value_channels < 1 ||
            columns != weight.rows || !cache.valid_paged() || capacity < 1 ||
            (!positions && !cache.sequences) ||
            (bias.data && (bias.type != input.type || bias.numel() != size_t(weight.rows))))
            throw std::invalid_argument("invalid cached projection geometry");
        if (query.overlaps(input) || query.overlaps(weight) || query.overlaps(bias))
            throw std::invalid_argument("cached projection output overlaps input");
    }
};
} // namespace tts
