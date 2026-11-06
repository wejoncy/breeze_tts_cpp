#pragma once
#include "core/tensor.h"
namespace tts {
struct SamplingRng {
    uint64_t seed = 0, offset = 0;
};
namespace ops {
// One sample per row, with an independent Philox state {seed, offset} per row.
// Probabilities are finite/nonnegative with a positive row sum. Outputs are I64;
// RNG storage is I64 [rows, 2]. The caller supplies all storage, also for replay.
void multinomial(TensorView probabilities, TensorView indices, TensorView random_states);
}
struct SamplingParams {
    float temperature = 0, top_p = 1;
    int top_k = 0;
    bool nucleus_before_temperature = false;
    bool include_threshold_token = true;
};
struct SamplingRow {
    int *codes = nullptr, *done = nullptr, *seen = nullptr;
    SamplingRng *rng = nullptr;
    int logits[3] = {}, branches = 1;
    float cfg = 1, cfg_ref = 1, cfg_ins = 1, penalty = 1;
    SamplingParams backbone, depth;
    const int *generated = nullptr;
    int eos_min_tokens = 0;
    bool eos_on_argmax = false;
};
struct SamplingLayout {
    int codebooks, entries, eos;
};
void sample( TensorView logits, SamplingLayout layout, int *codes,
            SamplingRng *rng, int batch, int branches, int codebook, float cfg, float temperature,
            int top_k, int *done, float top_p = 1.f, float cfg_ref = 1.f, float cfg_ins = 1.f,
            float penalty = 1.f, int *seen = nullptr, const SamplingRow *rows = nullptr);
inline void sample( TensorView logits, SamplingLayout layout,
                   const SamplingRow *rows, int batch, int codebook, int *done) {
    sample( logits, layout, nullptr, nullptr, batch, 1, codebook, 1, 0, 0, done, 1, 1, 1, 1,
           nullptr, rows);
}
} // namespace tts
