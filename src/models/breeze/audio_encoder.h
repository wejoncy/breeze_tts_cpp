#pragma once
#include "models/breeze/weights.h"
#include "models/breeze/audio_config.h"
namespace tts {
struct EncoderConv {
    TensorView weight, bias;
    int in, out, kernel, stride, dilation;
    bool replicate;
    EncoderConv(Weights &, const std::string &, int stride = 1, bool replicate = false,
                int dilation = 1);
    int rows(int input) const {
        return input / stride + (input % stride != 0);
    }
    void run(Backend &, TensorView x, TensorView projected, TensorView out, bool elu = false,
             TensorView residual = {}) const;
};
class AudioEncoder {
    struct Layer {
        TensorView pre, pre_bias, post, post_bias, qkv, out, fc1, fc2, attn_scale, mlp_scale;
    };
    struct Residual {
        EncoderConv first, second;
        Residual(Weights &, int index, int dilation);
    };
    struct Stage {
        std::vector<Residual> blocks;
        EncoderConv down;
        Stage(Weights &, int index, int ratio, const EncoderConfig &);
    };
    struct Storage;
    EncoderConfig config_;
    EncoderConv first_, last_;
    std::unique_ptr<EncoderConv> downsample_;
    std::vector<Stage> stages_;
    std::vector<Layer> layers_;
    TensorView semantic_proj_, acoustic_proj_;
    std::vector<TensorView> tables_, norms_;
    Tensor rope_storage_;
    TensorView rope_;

  public:
    AudioEncoder(Weights &, EncoderConfig);
    size_t workspace_bytes(int max_samples) const;
    std::vector<int> encode(Backend &, BufferAllocator &, const std::vector<float> &mono) const;
};
} // namespace tts
