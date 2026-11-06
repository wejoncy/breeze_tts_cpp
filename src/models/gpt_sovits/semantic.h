#pragma once
#include "layers/transformer.h"
#include "core/forward_context.h"
#include "io/json.h"

namespace tts {
class Semantic {
    TensorView phones_, audio_, positions_, phone_alpha_, audio_alpha_;
    Linear bert_, head_;
    TransformerEncoder transformer_;

  public:
    Semantic(Weights &, const std::string &prefix, const Json &, int max_context);
    int layers() const {
        return transformer_.layers();
    }
    int heads() const { return transformer_.heads(); }
    int hidden() const {
        return transformer_.hidden();
    }
    Type type() const {
        return transformer_.type();
    }
    TensorView encode_features(TensorView features, ForwardContext &context) const {
        return bert_.forward(features, context);
    }
    TensorView forward(TensorInputs, ForwardContext &) const;

};
} // namespace tts
