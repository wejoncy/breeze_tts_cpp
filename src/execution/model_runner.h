#pragma once
#include "execution/input_plan.h"
#include "execution/batch.h"
#include "execution/runner_resources.h"
#include "core/model.h"
#include "tts/model_info.h"
#include <exception>
#include <chrono>

namespace tts {
// Device execution and live request storage. Engine drives it through data
// records; it knows no scheduler, client queue or output callback.
class ModelRunner {
    struct Cache;
    struct Impl;
    std::unique_ptr<Impl> impl_;
    ForwardOutputs prefill_forward(const std::vector<Cache *> &);
    ForwardOutputs decode_forward();
    ForwardOutputs frame_outputs() const;
    void commit_frame(const SampledOutputs &);
    ForwardOutputs audio_forward();
    ForwardOutputs preparation_forward();
    ForwardOutputs decode_codes_forward();
    void commit_audio(const SampledOutputs &);
    void commit_preparation(const SampledOutputs &);
    void finish_step();
    std::shared_ptr<Cache> prepare(RequestPlan &, const ExecutionRow &);

  public:
    ModelRunner(const Model &, RunnerResources &, const InputDefinition &, const ModelInfo &,
                 size_t runtime_memory_bytes);
    ~ModelRunner();
    struct CacheUsage {
        size_t page_bytes, device_bytes, state_kv_bytes_per_row;
        int used_pages, backed_pages, state_rows;
    };
    // Initialization bounds for readbacks and per-row audio delivery.
    size_t max_step_samples() const;
    size_t max_readback_bytes() const;
    void set_engine_id(uint64_t);
    // Enqueue one successor while the current frame batch is still in flight.
    // EOS is collected afterwards; an extra frame for an ended row is discarded.
    // Engine retains the batch and passes it to forward() after collecting this one.
    bool submit_next(const ExecutionBatch &);
    bool can_submit_next() const;
    bool has_submitted() const;
    ForwardOutputs forward(ExecutionBatch);
    ForwardOutputs forward(PreparationBatch);
    ForwardOutputs forward_reference(const ReferencePlan &);
    ForwardOutputs forward_encode(const std::vector<float> &);
    // Cache survives between single-frame decode calls; transient workspaces do not.
    struct AudioDecodeCache;
    std::shared_ptr<AudioDecodeCache> prepare_codes(const std::vector<int> &);
    ForwardOutputs forward_codes(std::shared_ptr<AudioDecodeCache>);
    ForwardOutputs forward();
    void commit(const SampledOutputs &);
    bool finished() const;
    bool has_batch() const;
    bool preparing() const;
    bool has_control() const;
    Completion collect();
    PreparationBatch collect_preparation();
    HostResults collect_evaluation();
    // Empty until the last frame; relinquishes this quantum on every call.
    std::vector<float> collect_codes();
    // Engine has drained the stream after a failed control preparation. Preserve
    // resident generation states while dropping only the failed CPU/GPU task.
    void discard_control() noexcept;
    void abort();
    // CPU metadata only: Engine passes these requirements to Scheduler.
    ResourceRequirements requirements(const RequestPlan &) const;
    // Only drops request records after every execution row has been destroyed.
    bool release_if_idle(uint64_t request);

};
} // namespace tts
