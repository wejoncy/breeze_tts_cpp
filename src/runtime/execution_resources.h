#pragma once
#include "runtime/device.h"
#include <algorithm>
#include <map>
#include <memory>
namespace tts {
// Owns addresses borrowed by graphs. All growth and invalidation goes through reserve().
// The runner supplies its capture key and graph group; resources know neither model nor options.
template <class Key, class GraphState> class ExecutionResources {
    cudaStream_t stream_;
    DeviceArena arena_;
    std::map<Key, std::unique_ptr<GraphState>> graphs_;
    bool sealed_ = false;

  public:
    explicit ExecutionResources(DeviceContext &memory) : stream_(memory.stream), arena_(memory) {}
    void prepare(size_t bytes) {
        if (bytes <= arena_.capacity())
            return;
        if (sealed_)
            throw std::logic_error("execution workspace growth after initialization");
        CUDA(cudaStreamSynchronize(stream_));
        graphs_.clear();
        arena_.reserve(bytes);
    }
    BufferAllocator &workspace() {
        return arena_;
    }
    ArenaLease lease_workspace() {
        return ArenaLease(arena_);
    }
    void seal() {
        sealed_ = true;
    }
    GraphState *find_graphs(const Key &key) const {
        auto found = graphs_.find(key);
        return found == graphs_.end() ? nullptr : found->second.get();
    }
    GraphState &graphs_for(const Key &key) {
        auto found = graphs_.find(key);
        if (found != graphs_.end())
            return *found->second;
        if (sealed_)
            throw std::logic_error("graph creation after initialization");
        auto graphs = std::make_unique<GraphState>();
        auto &result = *graphs;
        graphs_.emplace(key, std::move(graphs));
        return result;
    }
};
} // namespace tts
