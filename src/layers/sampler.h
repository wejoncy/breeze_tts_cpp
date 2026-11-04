#pragma once
#include "core/forward_context.h"
#include "ops/sampling.h"

namespace tts {
class Sampler {
    SamplingLayout layout_;
  public:
    explicit Sampler(SamplingLayout layout) : layout_(layout) {}
    TensorView forward(TensorView logits, TensorView sampling_rows, TensorView done,
                        ForwardContext &context, int codebook_index = 0) const {
        if (context.mode == ForwardContext::Execute)
            sample(logits, layout_, sampling_rows.ptr<SamplingRow>(),
                   sampling_rows.rows, codebook_index, done.ptr<int>());
        return done;
    }
};
} // namespace tts
