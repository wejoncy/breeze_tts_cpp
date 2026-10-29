#pragma once
#include "transformer.h"

namespace tts {
// Every operation owns its streaming history. Weights remain immutable.
struct Conv {
    TensorView weight, bias;
    int in, out, kernel, dilation, stride;
    bool transpose, depthwise;
    Conv(Weights &, const std::string &, int dilation = 1, int stride = 1, bool transpose = false,
         bool depthwise = false);
    int history() const {
        return transpose ? (kernel - 1) / stride : (kernel - 1) * dilation;
    }
};
struct ConvState {
    Buffer history, projected;
    int batch, time;
    ConvState(const Conv &c, int b, int t, BufferAllocator *arena)
        : history(c.depthwise ? size_t(b) * c.history() * c.in * sizeof(float)
                              : size_t(b) *
                                    (t * (c.transpose ? c.stride : 1) +
                                     (c.transpose ? c.kernel - c.stride : c.history())) *
                                    c.out * sizeof(float),
                  arena),
          projected(c.depthwise ? 0 : size_t(b) * t * c.kernel * c.out * sizeof(float), arena),
          batch(b), time(t) {}
    void run(Backend &, const Conv &, TensorView, TensorView, const int *frame);
};
struct CodecOp {
    enum Kind { Convolution, Snake, Save, Add, LayerNorm, Linear, Gelu, ScaleAdd } kind;
    std::unique_ptr<Conv> conv;
    TensorView w, b;
    int save = 0;
};
struct CodecState {
    TransformerState transformer;
    std::vector<std::unique_ptr<ConvState>> conv;
    Buffer a, b, saves[2], rvq1, rvq2, latent, hidden, position, pcm;
    int batch, max_elements;
    CodecState(const class Codec &, int, BufferAllocator *arena);
    void reset(Backend &);
};
class Codec {
    std::vector<CodecOp> ops_;
    TensorView first_embedding_, rest_embedding_, first_proj_, rest_proj_, input_w_, input_b_,
        output_w_, output_b_;
    Conv pre_conv_;

  public:
    Transformer transformer;
    Codec(Weights &, TransformerConfig);
    TensorView forward(Backend &, CodecState &, const int *codes) const;
    friend struct CodecState;
};
} // namespace tts
