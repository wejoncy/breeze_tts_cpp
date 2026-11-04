#pragma once
#include "core/buffer.h"
#include "layers/transformer.h"
#include "io/json.h"

namespace tts {
class Bert {
    TensorView words_, positions_, segments_;
    AffineNorm embedding_norm_;
    TransformerEncoder encoder_;

  public:
    Bert(Weights &, const Json &, const std::string &prefix, int output_layers,
         bool quantize = false);
    int hidden() const { return encoder_.hidden(); }
    Type type() const { return encoder_.type(); }
    TensorView forward(TensorView input_ids, ForwardContext &) const;
};
} // namespace tts
