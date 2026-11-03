#pragma once
#include "execution/model_runner.h"

namespace tts {
class Backend;
}
namespace tts::gpt_sovits {
class Model;
class Runner final : public ModelRunner {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    Runner(const Model &, Backend &, int batch, int context, int page_tokens, size_t runtime_bytes);
    ~Runner() override;
    size_t max_step_samples() const override;
    void prepare_task(PreparationTask &) override;
    std::unique_ptr<RequestAdmission> admit(RequestPlan &) override;
    std::unique_ptr<RequestState> prepare(RequestPlan &, size_t, RequestAdmission &) override;
    size_t prefill_step(const std::vector<RequestState *> &) override;
    void step(const std::vector<RequestState *> &, const std::vector<AudioCallback> &) override;
    std::shared_ptr<const Reference> prepare_reference(ReferencePlan &) override;
};
} // namespace tts::gpt_sovits
