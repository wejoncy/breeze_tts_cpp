#pragma once
#include "kv_cache_view.h"
#include "packed_batch.h"
#include <cuda_runtime_api.h>
namespace tts {
void initialize_flash_attention();
void flash_prefill(cudaStream_t stream, TensorView q, TensorView k, TensorView v, TensorView out,
                   float *lse, int batch, int time, int slot, int capacity, int heads, int kvheads,
                   int dim, bool causal, int window, const PackedBatch *packed = nullptr,
                   CacheLayout cache = {});
void flash_decode(cudaStream_t stream, TensorView q, TensorView k, TensorView v, TensorView out,
                  float *lse, const int *positions, int batch, int heads, int kvheads,
                  CacheLayout cache);
} // namespace tts
