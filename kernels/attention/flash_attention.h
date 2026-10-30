#pragma once
#include "core/kv_cache_view.h"
#include "core/packed_batch.h"
#include <cuda_runtime_api.h>
namespace tts {
void initialize_flash_attention();
// Nonzero key_time includes a cached prefix; causal queries align to its end.
void flash_prefill(cudaStream_t stream, TensorView q, TensorView k, TensorView v, TensorView out,
                   float *lse, int batch, int time, int slot, int capacity, int heads, int kvheads,
                   int dim, bool causal, int window, const PackedBatch *packed = nullptr,
                   CacheLayout cache = {}, int key_time = 0);
void flash_decode(cudaStream_t stream, TensorView q, TensorView k, TensorView v, TensorView out,
                  float *lse, const int *positions, int batch, int heads, int kvheads,
                  CacheLayout cache);
} // namespace tts
