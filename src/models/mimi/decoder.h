#pragma once
#include "core/buffer.h"
#include "layers/transformer/decoder.h"
#include "models/mimi/config.h"
#include "ops/audio.h"
#include "layers/convolution.h"
#include "core/sequence_state.h"
#include <array>

namespace tts {
// Immutable codec weights; the runner supplies every request's cache/history.
class Codec {
    struct Operation {
        enum Kind { Convolution, Snake, Save, Add, LayerNorm, Linear, Gelu, ScaleAdd } kind;
        std::unique_ptr<CausalConvolution> conv;
        TensorView weight, bias;
        int save = 0;
    };
    struct Workspace {
        // Two live operands and two nested residual snapshots.
        std::array<Buffer, 4> activations;
        Buffer convolution, rvq_first, rvq_rest, latent, hidden, pcm;
        explicit Workspace(const Codec &, int batch);
    };
    std::vector<Operation> ops_;
    TensorView first_embedding_, rest_embedding_, first_proj_, rest_proj_, input_w_, input_b_,
        output_w_, output_b_;
    CausalConvolution pre_conv_;

  public:
    CodecConfig config;
    Transformer transformer;
    Codec(Weights &, const std::string &prefix, CodecConfig);
    void forward(TensorInputs, TensorOutputs, ForwardContext &) const;
    SequenceStateSpec history_spec() const;
};
} // namespace tts
