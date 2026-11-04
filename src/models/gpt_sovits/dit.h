#pragma once
#include "core/buffer.h"
#include "layers/convolution.h"
#include "layers/linear.h"
#include "models/gpt_sovits/sampling.h"
#include "io/json.h"
#include <map>

namespace tts {
class DiT {
    struct TextBlock {
        Convolution depthwise;
        TensorView norm, norm_bias, gamma, beta;
        Linear up, down;
    };
    struct Block {
        Linear modulation, qkv, out, up, down;
        Tensor time_modulation;
    };
    int hidden_, heads_, head_dim_, text_channels_, mel_channels_, steps_;
    Type type_;
    Linear time_in_, time_out_, text_projection_, condition_projection_, noise_projection_;
    Linear final_modulation_, output_;
    std::vector<TextBlock> text_blocks_;
    std::vector<Block> blocks_;
    std::vector<Convolution> position_convs_;
    TensorView text_positions_, rotary_;
    Tensor final_time_modulation_;
    std::map<int, int> sampling_offsets_;
    int quant_input_channels_ = 0;

  public:
    struct Condition {
        TensorView features;
        TensorView feature_rows;
        // Project independent prompt spans into the condition tensor. The row
        // indices select their projected rows; -1 is an unconditioned target.
        std::vector<TensorView> mel;
        TensorView mel_rows, retained_feature_rows;
    };
    DiT(Weights &, const std::string &prefix, const Json &, int max_time);
    Type type() const {
        return type_;
    }
    int steps() const {
        return steps_;
    }
    bool supports_steps(int steps) const { return sampling_offsets_.count(steps) != 0; }
    TensorView forward(const Condition &, ForwardContext &, bool negative) const;
    // Velocity may contain a second batch for the unconditional CFG branch.
    // Both branches consume the same noise directly; no concatenation is needed.
    TensorView forward(TensorView noisy_mel, TensorView condition, TensorView velocity,
                        ForwardContext &, int step, int steps) const;
};
} // namespace tts
