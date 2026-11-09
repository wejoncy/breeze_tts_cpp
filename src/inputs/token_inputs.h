#pragma once
#include "inputs/preparation.h"
#include "models/breeze/model_config.h"
#include "io/tokenizer.h"

namespace tts {
class TokenInputs final {
    Tokenizer tokenizer_;
    breeze::ModelConfig config_;
    int context_;
    InputDefinition definition_;
    GenerationSegment encode(const Input &, const GenerationOptions &, size_t index) const;
    InputBatch dummy(int batch, int time) const;
    class Preparation;

  public:
    TokenInputs(const std::string &directory, breeze::ModelConfig, int batch, int context);
    const InputDefinition &definition() const { return definition_; }
    std::unique_ptr<RequestPreparation> begin(const std::vector<Input> &, const GenerationOptions &) const;
    std::unique_ptr<ReferencePreparation> begin_reference(const std::vector<float> &, std::string,
        const std::string &) const;
    std::shared_ptr<const Reference> finish_reference(ReferencePlan &, HostResults) const;
};
} // namespace tts
