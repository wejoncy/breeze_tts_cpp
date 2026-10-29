#pragma once
#include "weights.h"
namespace tts {
struct EncoderConv {
    TensorView weight, bias;
    int in, out, kernel, stride;
    bool replicate;
    EncoderConv(Weights &, const std::string &, int stride = 1, bool replicate = false);
    void run(Backend &, TensorView x, TensorView projected, TensorView out, bool elu = false,
             TensorView residual = {}) const;
};
class AudioEncoder {
    struct Layer {
        TensorView pre, pre_bias, post, post_bias, qkv, out, fc1, fc2, attn_scale, mlp_scale;
    };
    struct Stage {
        EncoderConv first, second, down;
        Stage(Weights &, int index, int ratio);
    };
    EncoderConv first_, last_, downsample_;
    std::vector<Stage> stages_;
    std::vector<Layer> layers_;
    TensorView semantic_proj_, acoustic_proj_;
    std::vector<TensorView> tables_, norms_;
    Tensor rope_storage_;
    TensorView rope_;

  public:
    AudioEncoder(Weights &, int max_positions);
    static size_t workspace_bytes(int max_samples);
    std::vector<int> encode(Backend &, BufferAllocator &, const std::vector<float> &mono) const;
};
} // namespace tts
