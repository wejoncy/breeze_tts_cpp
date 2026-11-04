#pragma once
#include "models/mimi/decoder.h"
#include "models/mimi/encoder.h"
#include "models/breeze/model_config.h"
#include "core/model.h"
#include "layers/sampler.h"
#include "layers/transformer/codebook_decoder.h"

namespace tts::breeze {
// Construct once; runners borrow a const Model and own all mutable request state.
struct Model final : tts::Model {
    ModelConfig config;
    Transformer text, backbone, depth;
    Codec codec;
    AudioEncoder encoder;
    Sampler sampler;
    TensorView text_embedding, eoi, text_proj, audio_embedding, depth_proj, lm_head;
    Tensor depth_embedding_storage;
    TensorView depth_embedding;
    std::vector<TensorView> depth_heads;
    std::unique_ptr<ResidualCodebooks> residual_codebooks;
    Model(Weights &parameters, ModelConfig configuration);
    void forward(std::string_view program, TensorInputs inputs, TensorOutputs outputs,
                  ForwardContext &context) const override;
};
} // namespace tts::breeze
