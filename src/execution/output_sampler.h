#pragma once
#include "execution/forward_output.h"
#include "core/alignment.h"
#include "runtime/device.h"
#include "ops/cuda/cuda_check.h"
#include "support/profiler.h"
#include <cuda_runtime.h>

namespace tts {
// Collects the graph's sampled tokens/EOS or rendered PCM. Its pinned storage
// and event are independent of scheduling, model state and client delivery.
// Breeze's dependent codebook sampling remains fused with forward in the graph.
class OutputSampler {
    DeviceContext &memory_;
    cudaEvent_t ready_ = nullptr;
    size_t capacity_;
    Buffer host_;
    SampledOutputs sampled_;
    size_t transfer_checkpoint_ = 0;
    bool pending_ = false;
    bool recorded_ = false;
  public:
    OutputSampler(DeviceContext &memory, size_t bytes)
        : memory_(memory), capacity_(aligned_storage_bytes(bytes)),
          host_(capacity_, memory.host_allocator()) {
        CUDA(cudaSetDevice(memory_.device));
        CUDA(cudaEventCreateWithFlags(&ready_, cudaEventDisableTiming));
    }
    ~OutputSampler() {
        cudaSetDevice(memory_.device);
        if (pending_) {
            if (recorded_) cudaEventSynchronize(ready_);
            else { try { memory_.sync(); } catch (...) {} }
        }
        cudaEventDestroy(ready_);
    }
    OutputSampler(const OutputSampler &) = delete;
    void sample(const ForwardOutputs &outputs) {
        Range trace("tts.sample.enqueue");
        if (pending_) throw std::logic_error("sample output is still in flight");
        sampled_.values.clear();
        recorded_ = false;
        pending_ = true;
        size_t offset = 0;
        bool copied = false;
        for (const auto &field : outputs.values) {
            const auto source = field.value;
            const size_t bytes = source.size();
            if (!source.dense() || source.device != memory_.device || bytes > capacity_ - offset)
                throw std::logic_error("forward readback exceeds initialized sampling storage");
            auto destination = host_.view(source.rows, source.cols, source.type, offset);
            sampled_.values.push_back({field.name, destination});
            destination.copy_(source, true);
            copied |= bytes != 0;
            offset += aligned_storage_bytes(bytes);
            if (offset > capacity_) throw std::logic_error("sampling storage alignment overflow");
        }
        // Mark pending before recording so a failed record still retains DMA
        // backing; Engine drains the stream before destroying this sampler.
        // Empty control stages still need a compute completion checkpoint.
        if (!copied) memory_.copy_after_compute();
        transfer_checkpoint_ = memory_.transfer_checkpoint();
        CUDA(cudaEventRecord(ready_, memory_.copy_stream));
        recorded_ = true;
    }
    bool pending() const { return pending_; }
    bool ready() const {
        if (!pending_) return true;
        if (!recorded_) throw std::logic_error("sampling event submission failed");
        auto status = cudaEventQuery(ready_);
        if (status == cudaErrorNotReady) return false;
        CUDA(status);
        return true;
    }
    // Views remain valid until the next sample. Engine must commit them before
    // allowing the runner to retire the corresponding output-name storage.
    const SampledOutputs &collect() {
        Range trace("tts.sample.collect");
        if (!pending_ || !ready()) throw std::logic_error("sample output is not ready");
        memory_.complete_transfers(transfer_checkpoint_);
        pending_ = false;
        return sampled_;
    }
    // Only after the owner drains both streams on shutdown or device failure.
    void discard_after_drain() noexcept { pending_ = false; }
};
} // namespace tts
