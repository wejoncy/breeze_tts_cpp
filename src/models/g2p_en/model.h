#pragma once
#include "layers/recurrent.h"
#include "io/weights.h"

namespace tts {
class GraphemeToPhoneme {
    TensorView encoder_embedding_, decoder_embedding_, output_, output_bias_;
    GRUCell encoder_, decoder_;
    int encoder_end_, decoder_start_, decoder_end_, max_steps_;
  public:
    GraphemeToPhoneme(Weights &, const Json &metadata);
    // CPU packed character IDs and cu_seqlens; lengths receives emitted phones.
    Tensor forward(TensorView ids, TensorView cu_seqlens, TensorView lengths) const;
};
} // namespace tts
