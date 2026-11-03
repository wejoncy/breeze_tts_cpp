#pragma once
#include "execution/model_runner.h"
#include "runtime/graph_shapes.h"
#include "runtime/request_slots.h"

namespace tts {
class Backend;
class PagedKVCache;
class RequestAdmissionPool;
}
namespace tts::breeze {
struct Model;
struct GenerationState;
class Runner final : public ModelRunner, public AudioCodec {
    const Model &model;
    Backend &gpu;
    int max_batch, context;
    GraphShapes graph_shapes_;
    uint64_t request_sequence_ = 0;
    std::unique_ptr<PagedKVCache> kv_;
    std::unique_ptr<RequestAdmissionPool> admission_;
    RequestSlots slot_pool_;
    struct Slot;
    struct State;
    struct Admission;
    std::vector<std::unique_ptr<Slot>> slots_;
    struct Batch;
    std::unique_ptr<Batch> batch_;
    struct PrefillWork;
    std::unique_ptr<PrefillWork> prefill_;
    State &state(RequestState &);
    void initialize_graphs();

  public:
    Runner(const Model &, Backend &, int batch, int context, int page_tokens,
           size_t runtime_memory_bytes = 0);
    ~Runner() override;
    size_t max_step_samples() const override;
    std::unique_ptr<RequestAdmission> admit(RequestPlan &) override;
    std::unique_ptr<RequestState> prepare(RequestPlan &, size_t lane, RequestAdmission &) override;
    size_t prefill_step(const std::vector<RequestState *> &) override;
    void step(const std::vector<RequestState *> &, const std::vector<AudioCallback> &) override;
    std::shared_ptr<const Reference> prepare_reference(ReferencePlan &) override;
    std::vector<int> encode_audio(const std::vector<float> &) override;
    std::vector<float> decode_codes(const std::vector<int> &) override;
};
} // namespace tts::breeze
