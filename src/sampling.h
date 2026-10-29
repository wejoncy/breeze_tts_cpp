#pragma once
#include "tensor.h"
#include <cuda_runtime_api.h>
namespace tts {
void sample(cudaStream_t stream, TensorView logits, int *codes, uint64_t *rng, int batch,
            int branches, int codebook, float cfg, float temperature, int top_k, int *done,
            float top_p = 1.f, float cfg_ref = 1.f, float cfg_ins = 1.f, float penalty = 1.f,
            int *seen = nullptr);
} // namespace tts
