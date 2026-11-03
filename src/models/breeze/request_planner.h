#pragma once
#include "models/breeze/prepared_batch.h"
#include "execution/request_planner.h"

namespace tts::breeze {
// CPU data transferred from the scheduler to the executor. Validation and
// tokenization complete for every input before any input can execute.
struct Plan final : RequestPlan {
    GenerationOptions options;
    std::vector<PreparedBatch> inputs;
};
struct ReferenceInput final : ReferencePlan {
    std::vector<float> mono;
    std::string transcript;
};
class Planner final : public RequestPlanner {
    const Tokenizer &tokenizer_;
    const ModelConfig &config_;
    int max_batch_, context_;

  public:
    Planner(const Tokenizer &tokenizer, const ModelConfig &config, int batch, int context)
        : tokenizer_(tokenizer), config_(config), max_batch_(batch), context_(context) {}
    std::unique_ptr<RequestPreparation> begin(const std::vector<Input> &,
                                             const GenerationOptions &) const override;
    std::unique_ptr<ReferencePlan> prepare_reference(const std::vector<float> &,
                                                   std::string transcript,
                                                   const PreparationExecutor &execute = {},
                                                   const std::string &language = {}) const override;
};
} // namespace tts::breeze
