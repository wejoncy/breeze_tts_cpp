#pragma once
#include "layers/convolution.h"
#include "layers/linear.h"
#include "layers/diffusion/sampling.h"
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

  public:
    struct Workspace {
        Linear::Workspace linear;
        Buffer condition, negative_condition, x, normalized, update, qkv, attended, expanded;
        Buffer text, text_update, text_expanded, statistics, convolution, lse;
        int max_batch, max_time;
        Workspace(const DiT &, int batch, int time, BufferAllocator *);
    };
    struct Condition {
        TensorView features;
        const void *const *feature_rows = nullptr;
        // Project independent prompt spans into the shared workspace. The row
        // indices select their projected rows; -1 is an unconditioned target.
        std::vector<TensorView> mel;
        const int *mel_rows = nullptr;
        void *const *retained_feature_rows = nullptr;
    };
    DiT(Weights &, const std::string &prefix, const Json &, int max_time);
    Type type() const {
        return type_;
    }
    int steps() const {
        return steps_;
    }
    bool supports_steps(int steps) const { return sampling_offsets_.count(steps) != 0; }
    void prepare(Workspace &, TensorView mel_input, TensorView mel_output,
                 int max_rows) const;
    void prepare_condition(Workspace &, const Condition &, const ForwardBatch &,
                           bool negative) const;
    // Velocity may contain a second batch for the unconditional CFG branch.
    // Both branches consume the same noise directly; no concatenation is needed.
    void forward(Workspace &, TensorView noisy_mel, TensorView velocity, const ForwardBatch &,
                 int step, int steps) const;
};
} // namespace tts
