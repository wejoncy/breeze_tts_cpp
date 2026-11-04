#pragma once
#include "core/buffer.h"
#include "layers/convolution.h"
#include "layers/transformer.h"
#include "io/json.h"

namespace tts {
class Hubert {
    std::vector<Convolution> features_;
    TensorView temporal_weight_, temporal_bias_;
    AffineNorm feature_norm_, input_norm_;
    Linear projection_;
    Convolution position_;
    TransformerEncoder encoder_;

  public:
    Hubert(Weights &, const std::string &prefix, const Json &);
    int frames(int samples) const;
    TensorView forward(TensorView waveform, ForwardContext &) const;
};
} // namespace tts
