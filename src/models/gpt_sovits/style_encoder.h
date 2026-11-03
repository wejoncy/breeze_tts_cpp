#pragma once
#include "layers/convolution.h"
#include "layers/linear.h"
namespace tts::gpt_sovits {
class StyleEncoder {
    Linear first_, second_, qkv_, out_, final_;
    std::vector<Convolution> temporal_;
    int heads_;

  public:
    struct Workspace {
        Buffer hidden, gates, projected, attended, update, embedding, convolution, lse;
        int max_batch, max_time;
        Workspace(const StyleEncoder &, int batch, int time, BufferAllocator *);
    };
    StyleEncoder(Weights &, int heads, int temporal_layers);
    void forward(Workspace &, TensorView spectrum, TensorView style, const ForwardBatch &) const;
};
} // namespace tts::gpt_sovits
