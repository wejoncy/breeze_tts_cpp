#pragma once
#include "layers/transformer.h"
#include "io/json.h"

namespace tts::gpt_sovits {
class Semantic {
    TensorView phones_, audio_, positions_, phone_alpha_, audio_alpha_;
    Linear bert_, head_;
    TransformerEncoder transformer_;

  public:
    struct Workspace {
        TransformerEncoder::Workspace transformer;
        Buffer hidden, bert_projection, last;
        Workspace(const Semantic &s, int batch, int time, BufferAllocator *allocator)
            : transformer(s.transformer_, batch, time, allocator),
              hidden(size_t(batch) * time * s.transformer_.hidden() * bytes(s.transformer_.type()),
                     allocator),
              bert_projection(hidden.size(), allocator),
              last(size_t(batch) * s.hidden() * bytes(s.type()), allocator) {}
    };
    struct PrefillBatch {
        ForwardBatch context;
        const int *phones, *codes;
        TensorView encoded_features;
        ops::EmbeddingFusion phone_rows, audio_rows;
    };
    Semantic(Weights &, const Json &, int max_context);
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
    void prepare(int max_rows) const { transformer_.prepare(max_rows); }
    TensorView encode_features(Workspace &, TensorView) const;
    void prefill(Workspace &, const PrefillBatch &, TensorView logits) const;
    void decode(Workspace &, const int *codes, const int *audio_positions,
                const ForwardBatch &, TensorView logits,
                const int *const *code_rows = nullptr) const;
};
} // namespace tts::gpt_sovits
