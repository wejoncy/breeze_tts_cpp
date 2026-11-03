#pragma once
#include "layers/convolution.h"
#include "layers/transformer.h"
namespace tts::gpt_sovits {
class RelativeEncoder {
    struct Block {
        Linear qkv, out;
        TensorView relative_k, relative_v;
        Convolution first, second;
        AffineNorm attention_norm, feedforward_norm;
        int window;
    };
    std::vector<Block> blocks_;
    int hidden_, heads_;
    Type type_;

  public:
    struct Workspace {
        Buffer projected, attended, update, expanded, scratch;
        int max_batch, max_time, max_rows;
        Workspace(const RelativeEncoder &, int batch, int time, BufferAllocator *);
    };
    RelativeEncoder(Weights &, const std::string &, int layers, int heads);
    void prepare(Workspace &, int rows) const;
    void forward(Workspace &, TensorView x, const ForwardBatch &) const;
};
} // namespace tts::gpt_sovits
