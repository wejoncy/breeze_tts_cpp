#pragma once
#include "models/breeze/codec.h"
#include "models/breeze/audio_encoder.h"
#include "models/breeze/model_config.h"
#include "io/tokenizer.h"
#include "execution/model_runtime.h"
#include "ops/sampling.h"

namespace tts::breeze {
struct GenerationState;
struct DecodeBatch {
    int lanes;
    const SamplingRow *sampling;
    const int *const *codes;
    const int *row_request;
    const CacheSequence *cache;
};
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
    Model(Backend &, const ModelLoader &, ModelConfig);
    void depth_frame(GenerationState &, int batch, const DecodeBatch &,
                     bool debug = false) const;
    void next_frame(GenerationState &, int batch, const DecodeBatch &) const;
};
std::unique_ptr<ModelRuntime> load_model(const ModelLoader &, const ModelOptions &);
} // namespace tts::breeze
