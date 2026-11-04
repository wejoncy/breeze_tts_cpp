#pragma once
#include "layers/transformer.h"
#include "io/json.h"

namespace tts {
class Bert {
    TensorView words_, positions_, segments_;
    AffineNorm embedding_norm_;
    TransformerEncoder encoder_;

  public:
    struct Workspace {
        TransformerEncoder::Workspace encoder;
        Buffer features;
        Workspace(const Bert &b, int batch, int time, BufferAllocator *allocator,
                  int rows_capacity = 0)
            : encoder(b.encoder_, batch, time, allocator, rows_capacity),
              features(size_t(encoder.max_rows) * b.encoder_.hidden() * bytes(b.encoder_.type()),
                       allocator) {}
    };
    Bert(Weights &, const Json &, const std::string &prefix, int output_layers,
         bool quantize = false);
    int hidden() const { return encoder_.hidden(); }
    Type type() const { return encoder_.type(); }
    void prepare(int max_rows) const { encoder_.prepare(max_rows); }
    TensorView forward(Workspace &, const int *ids, const ForwardBatch &) const;
    TensorView forward(TensorView input_ids, ForwardContext &) const;
};
} // namespace tts
