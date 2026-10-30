#include "execution/request_planner.h"

namespace tts {
std::unique_ptr<RequestPlan> RequestPlanner::prepare(const std::vector<Input> &inputs,
                                                     const GenerationOptions &options) const {
    if (inputs.empty())
        throw std::invalid_argument("empty request batch");
    auto plan = std::make_unique<RequestPlan>();
    plan->options = options;
    plan->inputs.reserve(inputs.size());
    for (const auto &input : inputs)
        plan->inputs.push_back(
            prepare_batch(tokenizer_, config_, max_batch_, context_, {input}, options));
    return plan;
}
} // namespace tts
