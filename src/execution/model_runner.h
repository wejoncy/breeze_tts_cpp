#pragma once
#include "execution/request_planner.h"
#include <exception>

namespace tts {
// The executor exclusively owns this interface and every admission/state created
// through it. Model weights and init-time workspaces outlive those states.
class ModelRunner {
  public:
    // Reserves a request's capacity across all input waves. Destroy states before
    // releasing this admission, including on cancellation and executor failure.
    struct RequestAdmission {
        virtual ~RequestAdmission() = default;
    };
    // Opaque model state. No KV, tensor, codec, graph or architecture assumptions
    // cross this boundary. Destruction returns its leases to runtime pools.
    class RequestState {
      public:
        virtual ~RequestState() = default;
        virtual bool done() const = 0;
        virtual bool prefilled() const = 0;
        virtual Stats stats() const = 0;
        virtual std::exception_ptr error() const = 0;
    };
    virtual ~ModelRunner() = default;
    // Init-time host staging bound for all audio callbacks of one input in one
    // step(), including any pending tail and inter-segment silence.
    virtual size_t max_step_samples() const = 0;
    virtual void prepare_task(PreparationTask &) {
        throw std::invalid_argument("model does not define GPU preparation tasks");
    }
    // Empty means temporarily unavailable; retry after a release checkpoint.
    // Invalid requests and requests that cannot fit alone throw a terminal error.
    virtual std::unique_ptr<RequestAdmission> admit(RequestPlan &) = 0;
    // Empty means this admitted request has no free row credit. The input must
    // remain untouched; retry this row after a resource-release checkpoint.
    virtual std::unique_ptr<RequestState> prepare(RequestPlan &, size_t input,
                                                RequestAdmission &) = 0;
    // Advance model-defined preparation checkpoints. A model may pack fewer rows
    // than offered to respect admitted workspace; it must make forward progress.
    virtual size_t prefill_step(const std::vector<RequestState *> &) = 0;
    // Advance each ready input by a model-defined execution quantum. Callback i
    // receives that input's audio, if produced; state records completion/errors.
    virtual void step(const std::vector<RequestState *> &,
                      const std::vector<AudioCallback> &) = 0;
    virtual std::shared_ptr<const Reference> prepare_reference(ReferencePlan &) = 0;
};
// Optional discrete-code import/export capability. Acoustic or diffusion models
// need not implement a codec API to participate in request execution.
class AudioCodec {
  public:
    virtual ~AudioCodec() = default;
    virtual std::vector<int> encode_audio(const std::vector<float> &) = 0;
    virtual std::vector<float> decode_codes(const std::vector<int> &) = 0;
};
} // namespace tts
