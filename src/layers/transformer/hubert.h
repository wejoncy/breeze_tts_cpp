#pragma once
#include "layers/convolution.h"
#include "layers/transformer.h"
#include "layers/bert.h"
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
    struct Workspace {
        TransformerEncoder::Workspace encoder;
        Buffer convolution, activations[2], projected, statistics;
        int max_batch, max_samples;
        Workspace(const Hubert &, int batch, int samples, BufferAllocator *);
    };
    Hubert(Weights &, const std::string &prefix, const Json &);
    int frames(int samples) const;
    void prepare(Workspace &, int max_samples) const;
    TensorView forward(Workspace &, TensorView waveform, const ForwardBatch &) const;
};
} // namespace tts
