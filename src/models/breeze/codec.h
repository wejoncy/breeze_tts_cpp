#pragma once
#include "models/breeze/transformer.h"
#include "models/breeze/audio_config.h"
#include "kernels/audio/audio_kernels.h"

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
    size_t history_bytes(int b, int t) const {
        return depthwise ? size_t(b) * history() * in * sizeof(float)
                         : size_t(b) *
                               (t * (transpose ? stride : 1) +
                                (transpose ? kernel - stride : history())) *
                               out * sizeof(float);
    }
};
struct ConvState {
    Buffer history, projected;
    int batch, time;
    ConvState(const Conv &c, int b, int t, BufferAllocator *arena, bool persistent = true)
        : history(persistent ? c.history_bytes(b, t) : 0, arena),
          projected(c.depthwise ? 0 : size_t(b) * t * c.kernel * c.out * sizeof(float), arena),
          batch(b), time(t) {}
    void run(Backend &, const Conv &, TensorView, TensorView, const int *frame, int rows = 0,
             const ConvHistory *states = nullptr);
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
    CodecState(const class Codec &, int, BufferAllocator *arena, bool mapped = false);
    void reset(Backend &);
};
// Request persistence only; all convolution/transformer compute workspace is shared.
struct CodecHistory {
    Buffer keys, values, position;
    std::vector<Buffer> conv;
    CodecHistory(const class Codec &, BufferAllocator *);
    void reset(Backend &);
};
struct CodecBatch {
    const int *const *codes;
    const CacheSequence *cache;
    const ConvHistory *histories;
    int stride;
};
class Codec {
    std::vector<CodecOp> ops_;
    TensorView first_embedding_, rest_embedding_, first_proj_, rest_proj_, input_w_, input_b_,
        output_w_, output_b_;
    Conv pre_conv_;

  public:
    CodecConfig config;
    Transformer transformer;
    Codec(Weights &, CodecConfig);
    TensorView forward(Backend &, CodecState &, const int *codes, int batch = 0,
                       const CodecBatch *mapped = nullptr) const;
    friend struct CodecState;
    friend struct CodecHistory;
};
} // namespace tts
