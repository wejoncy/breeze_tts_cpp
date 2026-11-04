#pragma once
#include "layers/convolution.h"
#include "core/forward_context.h"
#include "models/mimi/config.h"
namespace tts {
class AudioEncoder {
    struct Layer {
        TensorView pre, pre_bias, post, post_bias, qkv, out, fc1, fc2, attn_scale, mlp_scale;
    };
    struct Residual {
        CausalConvolution first, second;
        Residual(Weights &, const std::string &prefix, int index, int dilation);
    };
    struct Stage {
        std::vector<Residual> blocks;
        CausalConvolution down;
        Stage(Weights &, const std::string &prefix, int index, int ratio, const EncoderConfig &);
    };
    struct Storage;
    EncoderConfig config_;
    CausalConvolution first_, last_;
    std::unique_ptr<CausalConvolution> downsample_;
    std::vector<Stage> stages_;
    std::vector<Layer> layers_;
    TensorView semantic_proj_, acoustic_proj_;
    std::vector<TensorView> tables_, norms_;
    Tensor rope_storage_;
    TensorView rope_;

  public:
    AudioEncoder(Weights &, const std::string &prefix, EncoderConfig);
    TensorView forward(TensorView waveform, ForwardContext &) const;
  private:
    size_t workspace_bytes(int max_samples) const;
    void prepare(ForwardContext &, int max_samples) const;
};
} // namespace tts
