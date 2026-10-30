#pragma once
#include "core/tensor.h"
#include <cuda_runtime_api.h>
namespace tts {
struct SamplingParams {
    float temperature = 0, top_p = 1;
    int top_k = 0;
};
struct SamplingRow {
    int *codes = nullptr, *done = nullptr, *seen = nullptr;
    uint64_t *rng = nullptr;
    int logits[3] = {}, branches = 1;
    float cfg = 1, cfg_ref = 1, cfg_ins = 1, penalty = 1;
    SamplingParams backbone, depth;
};
struct SamplingLayout {
    int codebooks, entries, eos;
};
void sample(cudaStream_t stream, TensorView logits, SamplingLayout layout, int *codes,
            uint64_t *rng, int batch, int branches, int codebook, float cfg, float temperature,
            int top_k, int *done, float top_p = 1.f, float cfg_ref = 1.f, float cfg_ins = 1.f,
            float penalty = 1.f, int *seen = nullptr, const SamplingRow *rows = nullptr);
inline void sample(cudaStream_t stream, TensorView logits, SamplingLayout layout,
                   const SamplingRow *rows, int batch, int codebook, int *done) {
    sample(stream, logits, layout, nullptr, nullptr, batch, 1, codebook, 1, 0, 0, done, 1, 1, 1, 1,
           nullptr, rows);
}
} // namespace tts
