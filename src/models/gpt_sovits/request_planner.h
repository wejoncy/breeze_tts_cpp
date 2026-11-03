#pragma once
#include "execution/request_planner.h"
#include "core/tensor.h"
#include "models/gpt_sovits/frontend.h"
#include "models/gpt_sovits/acoustic_rate.h"
#include "models/gpt_sovits/flow_sampling.h"
#include "io/json.h"
#include "models/gpt_sovits/synthesis_windows.h"

namespace tts::gpt_sovits {
inline constexpr double maximum_fragment_interval = 10.; // Seconds between text segments.
struct ReferenceData final : ReferenceFeatures {
    TextInput text;
    Type type;
    std::vector<int> codes;
    std::vector<uint16_t> style, mel, features, bert_projection, waveform_condition;
    int frames = 0;
};
struct PlannedInput {
    TextInput text;
    std::shared_ptr<const ReferenceData> reference;
    int capacity = 0, max_codes = 0;
    std::vector<TextInput> segments;
    int pause_frames = 0;
    SynthesisWindows windows;
};
struct Plan final : RequestPlan {
    GenerationOptions options;
    std::vector<PlannedInput> inputs;
};
struct ReferenceInput final : ReferencePlan {
    TextInput text;
    std::string transcript;
    std::vector<float> wave;
    std::vector<uint16_t> hubert_wave;
};
class Planner final : public RequestPlanner {
    class Preparation;
    Frontend frontend_;
    int context_, output_rate_, samples_per_frame_, reference_rate_, hubert_rate_;
    AcousticRate rate_;
    float min_seconds_, max_seconds_, silence_seconds_;
    Type type_;
    FlowSampling sampling_;
    WindowLimits windows_;

  public:
    Planner(const std::string &, const Json &, int context, Type);
    std::unique_ptr<RequestPreparation> begin(const std::vector<Input> &,
                                              const GenerationOptions &) const override;
    std::unique_ptr<ReferencePlan> prepare_reference(const std::vector<float> &,
                                                     std::string transcript,
                                                   const PreparationExecutor &execute = {},
                                                   const std::string &language = {}) const override;
};
} // namespace tts::gpt_sovits
