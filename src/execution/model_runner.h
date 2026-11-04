#pragma once
#include "execution/request_planner.h"
#include "execution/runner_resources.h"
#include "core/model.h"
#include "tts/model_info.h"
#include <exception>
#include <chrono>

namespace tts {
// The executor exclusively owns this interface and every admission/state created
// through it. Model weights and init-time workspaces outlive those states.
class ModelRunner {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    ModelRunner(const Model &, RunnerResources &, RequestPlanner &, const ModelInfo &,
                 size_t runtime_memory_bytes);
    // Reserves a request's capacity across all input waves. Destroy states before
    // releasing this admission, including on cancellation and executor failure.
    using RequestAdmission = RequestAdmissionPool::Lease;
    // Executor-owned progress and leases common to all networks. Tensor storage
    // stays in RunnerResources; a plan can add CPU preparation data.
    class RequestState {
      public:
        std::unique_ptr<RequestAdmission::RowLease> row_credit;
        std::unique_ptr<RequestSlots::Lease> slot;
        StatePool::Row &data;
        GenerationOptions options;
        std::unique_ptr<PagedKVCache::Lease> cache;
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        Stats metrics;
        std::exception_ptr failure;
        bool finished = false, is_prefilled = false;
        RequestState(RunnerResources &resources, std::unique_ptr<RequestAdmission::RowLease> credit,
                     std::unique_ptr<RequestSlots::Lease> reservation, GenerationOptions parameters)
            : row_credit(std::move(credit)), slot(std::move(reservation)),
              data(*resources.states.rows().at(slot->index)), options(std::move(parameters)) {}
        virtual ~RequestState() = default;
        bool done() const { return finished; }
        bool prefilled() const { return is_prefilled; }
        Stats stats() const { return metrics; }
        std::exception_ptr error() const { return failure; }
    };
    ~ModelRunner();
    // Init-time host staging bound for all audio callbacks of one input in one
    // step(), including any pending tail and inter-segment silence.
    size_t max_step_samples() const;
    void prepare_tasks(std::vector<PreparedTask> &);
    // Empty means temporarily unavailable; retry after a release checkpoint.
    // Invalid requests and requests that cannot fit alone throw a terminal error.
    std::unique_ptr<RequestAdmission> admit(RequestPlan &);
    // Empty means this admitted request has no free row credit. The input must
    // remain untouched; retry this row after a resource-release checkpoint.
    std::unique_ptr<RequestState> prepare(RequestPlan &, size_t input, RequestAdmission &);
    // Advance model-defined preparation checkpoints. A model may pack fewer rows
    // than offered to respect admitted workspace; it must make forward progress.
    size_t prefill_step(const std::vector<RequestState *> &);
    // Advance each ready input by a model-defined execution quantum. Callback i
    // receives that input's audio, if produced; state records completion/errors.
    void step(const std::vector<RequestState *> &, const std::vector<AudioCallback> &);
    std::shared_ptr<const Reference> prepare_reference(ReferencePlan &);
    std::vector<int> encode_audio(const std::vector<float> &);
    std::vector<float> decode_codes(const std::vector<int> &);
};
} // namespace tts
