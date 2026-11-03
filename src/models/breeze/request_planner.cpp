#include "models/breeze/request_planner.h"

namespace tts::breeze {
std::unique_ptr<RequestPreparation> Planner::begin(const std::vector<Input> &inputs,
                                                   const GenerationOptions &options) const {
    if (inputs.empty())
        throw std::invalid_argument("empty request batch");
    if (options.speed_factor != 1 || options.sample_steps || options.cfg_rate != -1)
        throw std::invalid_argument("Breeze does not support acoustic speed or diffusion controls");
    auto plan = std::make_unique<Plan>();
    plan->options = options;
    plan->inputs.reserve(inputs.size());
    for (const auto &input : inputs) {
        if (!input.text_split_method.empty() || input.fragment_interval != 0)
            throw std::invalid_argument("Breeze does not define text-segment synthesis controls");
        plan->inputs.push_back(
            prepare_batch(tokenizer_, config_, max_batch_, context_, {input}, options));
    }
    return std::make_unique<ImmediatePreparation>(std::move(plan));
}
std::unique_ptr<ReferencePlan> Planner::prepare_reference(const std::vector<float> &mono,
                                                        std::string transcript,
                                                        const PreparationExecutor &execute,
                                                        const std::string &language) const {
    if (mono.empty() || mono.size() > size_t(config_.max_reference_frames) * config_.samples_per_frame)
        throw std::invalid_argument("reference audio exceeds context or is empty");
    if (transcript.empty() || transcript.size() > 65536 || transcript.find('\0') != std::string::npos)
        throw std::invalid_argument("reference transcript is required (1..65536 bytes)");
    auto plan = std::make_unique<ReferenceInput>();
    plan->mono = mono;
    plan->transcript = std::move(transcript);
    return plan;
}
} // namespace tts::breeze
