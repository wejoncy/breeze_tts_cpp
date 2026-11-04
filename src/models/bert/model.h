#pragma once
#include "core/buffer.h"
#include "layers/transformer.h"
#include "io/json.h"

namespace tts {
class Bert {
    TensorView words_, positions_, segments_;
    AffineNorm embedding_norm_;
    TransformerEncoder encoder_;

    struct Workspace {
        TransformerEncoder::Workspace encoder;
        Buffer features;
        Workspace(const Bert &b, int batch, int time,
                  int rows_capacity = 0)
            : encoder(b.encoder_, batch, time, rows_capacity),
              features(size_t(encoder.max_rows) * b.encoder_.hidden() * bytes(b.encoder_.type())) {}
    };
    TensorView forward(Workspace &, const int *ids, const ForwardBatch &) const;

  public:
    Bert(Weights &, const Json &, const std::string &prefix, int output_layers,
         bool quantize = false);
    int hidden() const { return encoder_.hidden(); }
    Type type() const { return encoder_.type(); }
    TensorView forward(TensorView input_ids, ForwardContext &) const;
};
} // namespace tts
