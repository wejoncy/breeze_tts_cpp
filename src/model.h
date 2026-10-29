#pragma once
#include "codec.h"
#include "audio_encoder.h"
#include "model_config.h"
#include "tokenizer.h"

namespace tts {
// Construct once; runners borrow a const Model and own all mutable request state.
struct Model {
    ModelConfig config;
    Weights weights;
    Tokenizer tokenizer;
    Transformer text, backbone, depth;
    Codec codec;
    AudioEncoder encoder;
    TensorView text_embedding, eoi, text_proj, audio_embedding, depth_proj, lm_head;
    Tensor depth_embedding_storage;
    TensorView depth_embedding;
    std::vector<TensorView> depth_heads;
    Model(Backend &, const std::string &directory, ModelConfig);
};
} // namespace tts
