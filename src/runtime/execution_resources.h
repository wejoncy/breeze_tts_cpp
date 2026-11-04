#pragma once
#include "runtime/device.h"
#include "runtime/cuda_graph_runner.h"
namespace tts {
// Owns addresses borrowed by graphs. All growth and invalidation goes through reserve().
// Workspace and graph lifetime share one initialization boundary for every model.
class ExecutionResources {
    cudaStream_t stream_;
    DeviceArena arena_;
    CUDAGraphRunner graphs_;

  public:
    explicit ExecutionResources(DeviceContext &memory)
        : stream_(memory.stream), arena_(memory), graphs_(memory.stream) {}
    void prepare(size_t bytes) {
        if (bytes <= arena_.capacity())
            return;
        if (graphs_.sealed())
            throw std::logic_error("execution workspace growth after initialization");
        CUDA(cudaStreamSynchronize(stream_));
        graphs_.clear();
        arena_.reserve(bytes);
    }
    ScratchAllocator &workspace() {
        return arena_;
    }
    ScratchScope lease_workspace() {
        return ScratchScope(arena_);
    }
    void seal() {
        graphs_.seal();
    }
    CUDAGraphRunner &graphs() { return graphs_; }
    const CUDAGraphRunner &graphs() const { return graphs_; }
};
} // namespace tts
