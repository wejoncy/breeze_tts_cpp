#pragma once
#include "execution/input_plan.h"

namespace tts {
// CPU-owned continuation. Engine transfers GPU tasks by ownership.
class RequestPreparation {
  public:
    virtual ~RequestPreparation() = default;
    virtual std::unique_ptr<PreparationTask> next() = 0;
    virtual bool ready() const { return true; }
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
// CPU reference preparation yields model programs to Engine just like request
// preparation. It has no execution callback or dependency on a model runner.
class ReferencePreparation {
  public:
    virtual ~ReferencePreparation() = default;
    virtual std::unique_ptr<PreparationTask> next() = 0;
    virtual void accept(std::unique_ptr<PreparationTask>) {
        throw std::logic_error("unexpected reference preparation result");
    }
    virtual std::unique_ptr<ReferencePlan> finish() = 0;
};
class ImmediateReferencePreparation final : public ReferencePreparation {
    std::unique_ptr<ReferencePlan> plan_;
  public:
    explicit ImmediateReferencePreparation(std::unique_ptr<ReferencePlan> plan) : plan_(std::move(plan)) {}
    std::unique_ptr<PreparationTask> next() override { return {}; }
    std::unique_ptr<ReferencePlan> finish() override { return std::move(plan_); }
};
} // namespace tts
