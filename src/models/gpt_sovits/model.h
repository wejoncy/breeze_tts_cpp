#pragma once
#include "models/gpt_sovits/acoustic.h"
#include "models/gpt_sovits/flow.h"
#include "models/gpt_sovits/reference_encoder.h"
#include "models/gpt_sovits/semantic.h"
#include "models/gpt_sovits/vocoder.h"
#include "models/gpt_sovits/pronunciation.h"
#include "execution/model_runtime.h"

namespace tts::gpt_sovits {
// Network composition only. Scheduling, admission, KV pages, graph ownership
// and workspace storage belong to the common execution/runtime layer.
class Model {
  public:
    const Json config;
    Weights weights;
    Bert bert;
    Pronunciation pronunciation;
    Semantic semantic;
    ReferenceEncoder reference;
    Acoustic acoustic;
    Flow flow;
    Vocoder vocoder;
    const int window_frames, reference_frames, rolling_frames, target_frames;
    const int mel_channels, feature_channels;
    const float mel_minimum, mel_maximum;
    Model(Backend &, const ModelLoader &, int context);

    void prepare_flow(Flow::Workspace &, const DiT::Condition &, const ForwardBatch &,
                        const int *prefix_lengths, const uint64_t *seeds,
                        const uint64_t *windows, FlowSampling) const;
    void flow_step(Flow::Workspace &, const ForwardBatch &, const int *prefix_lengths,
                   int prefix, int step, FlowSampling,
                   const ops::EulerOutput &output = {}) const;
};
std::unique_ptr<ModelRuntime> load_model(const ModelLoader &, const ModelOptions &);
} // namespace tts::gpt_sovits
