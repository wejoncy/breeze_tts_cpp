#pragma once
#include "layers/bert.h"

namespace tts::gpt_sovits {
class Pronunciation {
    Bert bert_;
    Linear classifier_, pos_classifier_;
    TensorView descriptor_bias_, char_descriptor_, second_order_descriptor_, legal_;
    int max_tokens_;

  public:
    struct Workspace {
        Bert::Workspace bert;
        Buffer selected, logits, pos_logits;
        int max_time;
        Workspace(const Pronunciation &, int max_time, BufferAllocator *);
    };
    Pronunciation(Weights &, const Json &metadata);
    int max_tokens() const { return max_tokens_; }
    void prepare()const { bert_.prepare(max_tokens_); }
    // One sentence encoding serves all its pronunciation queries. Query arrays
    // contain padded_time entries; negative positions/character IDs are inactive.
    void forward(Workspace &, const int *ids, const int *query_positions,
                 const int *char_ids, const int *cu_seqlens, int padded_time,
                 int *output_ids, int batch, const int *token_positions) const;
};
} // namespace tts::gpt_sovits
