#pragma once
#include "layers/audio/acoustic.h"
#include "layers/diffusion/flow.h"
#include "layers/audio/reference_encoder.h"
#include "layers/transformer/conditioned_decoder.h"
#include "layers/audio/vocoder.h"
#include "layers/text/pronunciation.h"
#include "core/model.h"
#include "layers/sampler.h"

namespace tts::gpt_sovits {
// Network composition only. Scheduling, admission, KV pages, graph ownership
// and workspace storage belong to the common execution/runtime layer.
class Model final : public tts::Model {
    TensorView text_features(TensorInputs, ForwardContext &) const;
  public:
    const Json config;
    Bert bert;
    Pronunciation pronunciation;
    Semantic semantic;
    ReferenceEncoder reference;
    Acoustic acoustic;
    Flow flow;
    Vocoder vocoder;
    Sampler sampler;
    const int window_frames, reference_frames, rolling_frames, target_frames;
    const int mel_channels, feature_channels;
    const float mel_minimum, mel_maximum;
    Model(Weights &parameters, const Json &configuration, const Json &pronunciation_configuration,
           int context);
    void forward(std::string_view program, TensorInputs inputs, TensorOutputs outputs,
                  ForwardContext &context) const override;
};
} // namespace tts::gpt_sovits
