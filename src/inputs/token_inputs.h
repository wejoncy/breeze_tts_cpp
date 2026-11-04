#pragma once
#include "execution/request_planner.h"
#include "models/breeze/model_config.h"
#include "io/tokenizer.h"

namespace tts {
class TokenInputs final : public RequestPlanner {
    Tokenizer tokenizer_;
    breeze::ModelConfig config_;
    int context_;
    InputDefinition definition_;
    GenerationSegment encode(const Input &, const GenerationOptions &, size_t index) const;
    InputBatch dummy(int batch, int time) const;

  public:
    TokenInputs(const std::string &directory, breeze::ModelConfig, int batch, int context);
    const InputDefinition &definition() const override { return definition_; }
    std::unique_ptr<RequestPreparation> begin(const std::vector<Input> &, const GenerationOptions &) const override;
    std::unique_ptr<ReferencePlan> prepare_reference(const std::vector<float> &, std::string,
        const PreparationExecutor &, const std::string &) const override;
    std::shared_ptr<const Reference> finish_reference(ReferencePlan &, HostResults) const override;
};
} // namespace tts
