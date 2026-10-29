#pragma once
#include "graph.h"
#include "kv_cache.h"
#include "request.h"
#include <algorithm>
#include <map>
#include <tuple>
namespace tts {
// Owns addresses borrowed by graphs. All growth and invalidation goes through reserve().
class ExecutionResources {
    cudaStream_t stream_;
    DeviceArena arena_;
    PagedKVCache kv_;
    using Key = std::tuple<int, int, int, float, float, int, float, float, bool, int, float, float,
                           bool, float, float, uint64_t>;
    struct Entry {
        uint64_t used;
        std::unique_ptr<Graphs> graphs;
    };
    std::map<Key, Entry> graphs_;
    uint64_t clock_ = 0;
    void reserve(size_t bytes, int pages) {
        if (bytes <= arena_.capacity() && pages <= kv_.capacity())
            return;
        CUDA(cudaStreamSynchronize(stream_));
        graphs_.clear();
        kv_.reserve(pages);
        arena_.reserve(bytes);
    }

  public:
    ExecutionResources(DeviceContext &memory, int layers, int width, Type type, int lanes,
                       int context, int page_tokens)
        : stream_(memory.stream), arena_(memory),
          kv_(memory, layers, width, type, lanes, context, page_tokens) {}
    void prepare(size_t bytes, const std::vector<int> &capacities = {}) {
        reserve(bytes, capacities.empty() ? 0 : kv_.required_pages(capacities));
    }
    BufferAllocator &workspace() {
        return arena_;
    }
    ArenaLease lease_workspace() {
        return {arena_, stream_};
    }
    PagedKVCache::Lease lease_kv(const std::vector<int> &capacities) {
        return {kv_, stream_, capacities};
    }
    KVCacheView kv_view() const {
        return kv_.view();
    }
    Graphs &graphs_for(int batch, int branches, int rows, const GenerationOptions &o,
                       uint64_t revision) {
        Key key{batch,
                branches,
                rows,
                o.temperature,
                o.cfg_scale,
                o.top_k,
                o.top_p,
                o.repetition_penalty,
                o.do_sample,
                o.depth_top_k,
                o.depth_temperature,
                o.depth_top_p,
                o.depth_do_sample,
                o.cfg_scale_ref,
                o.cfg_scale_ins,
                revision};
        auto found = graphs_.find(key);
        if (found != graphs_.end()) {
            found->second.used = ++clock_;
            return *found->second.graphs;
        }
        constexpr size_t limit = 8;
        if (graphs_.size() >= limit) {
            auto oldest =
                std::min_element(graphs_.begin(), graphs_.end(), [](const auto &a, const auto &b) {
                    return a.second.used < b.second.used;
                });
            graphs_.erase(oldest);
        }
        auto graphs = std::make_unique<Graphs>();
        auto &result = *graphs;
        graphs_.emplace(key, Entry{++clock_, std::move(graphs)});
        return result;
    }
};
} // namespace tts
