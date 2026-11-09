#pragma once
#include "inputs/preparation.h"

namespace tts {
class ModelLoader;
// Model-specific CPU text and reference inputs, owned and called by Engine.
class PhonemeInputs final {
    struct Impl;
    std::unique_ptr<Impl> impl_;
  public:
    PhonemeInputs(const ModelLoader &, int batch, int context, Type);
    ~PhonemeInputs();
    const InputDefinition &definition() const;
    std::unique_ptr<RequestPreparation> begin(const std::vector<Input> &, const GenerationOptions &) const;
    std::unique_ptr<ReferencePreparation> begin_reference(const std::vector<float> &, std::string,
                                                        const std::string &) const;
    std::shared_ptr<const Reference> finish_reference(ReferencePlan &, HostResults) const;
};
} // namespace tts
