#pragma once
#include "core/forward_batch.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <vector>
namespace tts {
// Borrowed FP32 scratch, reserved from the shared backend pool at initialization.
struct FlashDecodeWorkspace {
    TensorView output, lse;
};
class FlashDecodePlan {
    int resident_slots_, key_capacity_, tiles_, limits_;
    std::vector<uint8_t> choices_;
    static int choose(int tiles, int limit) {
        int best = 1, best_work = tiles;
        for (int splits = 2; splits <= limit; ++splits) {
            int blocks = (tiles + splits - 1) / splits;
            if (int64_t(splits - 1) * blocks >= tiles) continue;
            // Source-work proxy inherited from the single/max policy, not fitted timings.
            int work = blocks + 3 + (splits - 1) / 8;
            if (work < best_work) { best = splits; best_work = work; }
        }
        return best;
    }
    int limit(int batch, int heads) const {
        if (batch < 1 || heads < 1)
            throw std::invalid_argument("invalid attention planning geometry");
        int64_t rows = int64_t(batch) * heads;
        if (rows > 65535) return 1;
        return int(std::min<int64_t>(limits_, std::max<int64_t>(1, resident_slots_ / rows)));
    }
  public:
    FlashDecodePlan(int resident_slots, int key_capacity)
        : resident_slots_(resident_slots), key_capacity_(key_capacity),
          tiles_(key_capacity > 0 ? (key_capacity - 1) / 64 + 1 : 0),
          limits_(std::min({128, tiles_, resident_slots})) {
        if (resident_slots < 1 || key_capacity < 1)
            throw std::invalid_argument("invalid attention plan capacity");
        if (size_t(tiles_) > SIZE_MAX / size_t(limits_))
            throw std::overflow_error("attention selection capacity");
        choices_.resize(size_t(limits_) * tiles_);
        // Bounded by kernel split capacity and K tiles, independent of maximum batch.
        for (int parallel = 1; parallel <= limits_; ++parallel)
            for (int tiles = 1; tiles <= tiles_; ++tiles)
                choices_[size_t(parallel - 1) * tiles_ + tiles - 1] =
                    uint8_t(choose(tiles, std::min(parallel, tiles)));
    }
    int select(int batch, int heads, int key_length) const {
        if (key_length < 1 || key_length > key_capacity_)
            throw std::invalid_argument("attention exceeds initialized key capacity");
        int tiles = (key_length - 1) / 64;
        return choices_[size_t(limit(batch, heads) - 1) * tiles_ + tiles];
    }
    std::vector<int> graph_splits(int batch, int heads) const {
        const auto *choices = choices_.data() + size_t(limit(batch, heads) - 1) * tiles_;
        std::vector<int> result;
        bool used[129]{};
        for (int tiles = 0; tiles < tiles_; ++tiles) used[choices[tiles]] = true;
        for (int splits = 1; splits <= limits_; ++splits)
            if (used[splits]) result.push_back(splits);
        return result;
    }
};
// Device metadata addresses subranges directly inside packed projections. Prefix
// and causal suffix attention can share storage without gathering padded rows.
struct DenseAttentionLayout {
    int query_batch_rows = 0, key_batch_rows = 0;
    const AttentionSequence *sequences = nullptr;
    const void *const *key_sources = nullptr, *const *value_sources = nullptr;
    const int *cu_seqlens_q = nullptr, *cu_seqlens_k = nullptr;
};
void initialize_flash_attention();
// Initialization only, for the exact compiled split-decode kernel.
int flash_decode_resident_blocks(int head_dim, Type);
// Nonzero key_time includes a cached prefix; causal queries align to its end.
void prefill_attention(cudaStream_t stream, TensorView q, TensorView k, TensorView v, TensorView out,
                   float *lse, int batch, int time, int slot, int capacity, int heads, int kvheads,
                   int dim, bool causal, int window, const SequenceLayout *packed = nullptr,
                   CacheLayout cache = {}, int key_time = 0, float softmax_scale = 0,
                   DenseAttentionLayout dense = {});
void dense_attention(cudaStream_t, TensorView q, TensorView k, TensorView v, TensorView out,
                 float *lse, int batch, int query_time, int key_time, int heads, int head_dim,
                 bool causal = false, float softmax_scale = 0, DenseAttentionLayout = {});
void decode_attention(cudaStream_t stream, TensorView q, TensorView k, TensorView v, TensorView out,
                  float *lse, const int *positions, int batch, int heads, int kvheads,
                  CacheLayout cache, FlashDecodeWorkspace scratch = {}, int partitions = 1);
} // namespace tts
