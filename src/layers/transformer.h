#pragma once
#include "core/buffer.h"
#include "layers/linear.h"
#include "ops/attention.h"

namespace tts {
struct AffineNorm {
    TensorView weight, bias;
    float epsilon = 1.e-5f;
};
struct EncoderBlock {
    Linear qkv, out, up, down;
    AffineNorm attention_norm, feedforward_norm;
    ops::Activation activation = ops::Activation::Gelu;
};
// Model files describe parameter names and layer order. This layer implements
// dense post-norm attention/MLP execution without checkpoint-family branches.
class TransformerEncoder {
    std::vector<EncoderBlock> blocks_;
    int hidden_, heads_;
    Linear::Workspace::Capacity quantization_capacity_;
    Type type_;

    struct Workspace {
        Linear::Workspace linear;
        Buffer projected, attended, update, expanded, lse;
        int max_batch, max_time, max_rows;
        Workspace(const TransformerEncoder &, int batch, int time,
                  int rows_capacity = 0);
    };
  public:
    TransformerEncoder(std::vector<EncoderBlock>, int heads);
    int hidden() const {
        return hidden_;
    }
    int heads() const {
        return heads_;
    }
    int layers() const {
        return int(blocks_.size());
    }
    Type type() const {
        return type_;
    }
    TensorView forward(TensorView hidden_states, ForwardContext &) const;

  private:
    void prepare(Workspace &, int max_rows) const;
    // A nonzero prefix has bidirectional queries followed by a causal suffix.
    void forward(Workspace &, TensorView x, const ForwardBatch &) const;
};
} // namespace tts
