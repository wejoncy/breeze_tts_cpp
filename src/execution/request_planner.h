#pragma once
#include "tts/request.h"

namespace tts {
// Model-defined preparation exchanged by ownership through the executor queue.
// CPU tokenization and GPU conditioning can alternate without sharing state.
struct PreparationTask {
    virtual ~PreparationTask() = default;
};
using PreparationExecutor =
    std::function<std::unique_ptr<PreparationTask>(std::unique_ptr<PreparationTask>)>;
// CPU-owned model input, transferred once from scheduler to executor. Its model
// defines token layout, auxiliary features and validation; the engine never reads it.
struct RequestPlan {
    virtual ~RequestPlan() = default;
};
struct ReferencePlan {
    virtual ~ReferencePlan() = default;
};
// CPU-owned continuation. GPU tasks cross the executor queue by ownership.
class RequestPreparation {
  public:
    virtual ~RequestPreparation() = default;
    virtual std::unique_ptr<PreparationTask> next() = 0;
    virtual void accept(std::unique_ptr<PreparationTask>) {
        throw std::logic_error("unexpected GPU preparation result");
    }
    virtual std::unique_ptr<RequestPlan> finish() = 0;
};
class ImmediatePreparation final : public RequestPreparation {
    std::unique_ptr<RequestPlan> plan_;
  public:
    explicit ImmediatePreparation(std::unique_ptr<RequestPlan> plan) : plan_(std::move(plan)) {}
    std::unique_ptr<PreparationTask> next() override { return {}; }
    std::unique_ptr<RequestPlan> finish() override { return std::move(plan_); }
};
class RequestPlanner {
  public:
    virtual ~RequestPlanner() = default;
    virtual std::unique_ptr<RequestPreparation> begin(const std::vector<Input> &,
                                                      const GenerationOptions &) const = 0;
    std::unique_ptr<RequestPlan> prepare(const std::vector<Input> &inputs,
                                        const GenerationOptions &options,
                                        const PreparationExecutor &execute = {}) const {
        auto preparation = begin(inputs, options);
        while (auto task = preparation->next()) {
            if (!execute) throw std::logic_error("model preparation requires the executor");
            preparation->accept(execute(std::move(task)));
        }
        return preparation->finish();
    }
    virtual std::unique_ptr<ReferencePlan> prepare_reference(const std::vector<float> &mono,
                                                           std::string transcript,
                                                           const PreparationExecutor & = {},
                                                           const std::string &language = {}) const = 0;
};
} // namespace tts
