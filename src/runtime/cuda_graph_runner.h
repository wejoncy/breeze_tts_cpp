#pragma once
#include "runtime/graph.h"
#include "core/forward_context.h"
#include <algorithm>
#include <initializer_list>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace tts {
// Whole forwards and borrowed compute calls share one shape registry.
// Capture is init-only; replay reads sealed shapes and stable addresses.
class CUDAGraphRunner {
  public:
    using Shape = std::vector<int>;

  private:
    struct ShapeLess {
        using is_transparent = void;
        template <class Left, class Right>
        bool operator()(const Left &left, const Right &right) const {
            return std::lexicographical_compare(left.begin(), left.end(), right.begin(), right.end());
        }
    };
    struct Entry {
        std::vector<const void *> buffers;
        Graph graph;
    };
    using Program = std::map<Shape, Entry, ShapeLess>;
    cudaStream_t stream_;
    std::map<const void *, std::map<std::string, Program, std::less<>>,
             std::less<const void *>> programs_;
    bool sealed_ = false;
    void initializing() const {
        if (sealed_) throw std::logic_error("CUDA graph mutation after initialization");
    }
    void ready() const {
        if (!sealed_) throw std::logic_error("CUDA graph execution before initialization");
    }
    const Program *program(const void *owner, const char *name) const {
        if (!name) return nullptr;
        auto scope = programs_.find(owner);
        if (scope == programs_.end()) return nullptr;
        auto found = scope->second.find(name);
        return found == scope->second.end() ? nullptr : &found->second;
    }
    const Entry *find(const char *name, std::initializer_list<int> shape,
                       const void *owner = nullptr) const {
        if (!name || !shape.size()) return nullptr;
        auto family = program(owner, name);
        if (!family) return nullptr;
        auto graph = family->find(shape);
        return graph == family->end() ? nullptr : &graph->second;
    }

  public:
    explicit CUDAGraphRunner(cudaStream_t stream) : stream_(stream) {}
    CUDAGraphRunner(const CUDAGraphRunner &) = delete;
    CUDAGraphRunner &operator=(const CUDAGraphRunner &) = delete;
    bool sealed() const { return sealed_; }
    void clear() {
        initializing();
        programs_.clear();
    }
    void register_shape(const char *name, std::initializer_list<int> shape) {
        initializing();
        if (!name || !*name || !shape.size())
            throw std::invalid_argument("invalid CUDA graph program/shape");
        programs_[nullptr][name].try_emplace(Shape(shape));
    }
    template <class Forward> void capture(const char *name, Forward &&forward) {
        initializing();
        if (!name || !*name)
            throw std::invalid_argument("invalid CUDA graph program");
        if (!program(nullptr, name))
            throw std::invalid_argument("CUDA graph program was not registered");
        auto &family = programs_.at(nullptr).at(name);
        GraphCaptureProgress progress(name, family.size());
        for (auto it = family.rbegin(); it != family.rend(); ++it) {
            it->second.graph.capture(stream_, [&] { forward(it->first); });
            progress.advance();
        }
    }
    void seal() {
        for (const auto &owner : programs_)
            for (const auto &program : owner.second) {
                for (const auto &entry : program.second)
                    if (!entry.second.graph.executable)
                        throw std::logic_error("CUDA graph program has an uncaptured shape");
                if (owner.first) TTS_LOG(TTS_LOG_INFO, "engine.init", "CUDA graph capture {} shapes={}",
                        program.first, program.second.size());
            }
        sealed_ = true;
    }
    // Calls borrow a closure only for capture. Replay retains addresses and
    // executable graphs, never layer workspaces or request-owned host objects.
    bool capture(const ForwardCall &call) {
        initializing();
        auto &program = programs_[call.owner][call.name];
        auto [region, inserted] = program.try_emplace(Shape(call.shape));
        if (inserted) {
            region->second.buffers.assign(call.buffers);
            region->second.graph.capture(stream_, [&] { call.invoke(call.arguments); });
        }
        return true;
    }
    bool run(const ForwardCall &call) const {
        ready();
        auto entry = find(call.name, call.shape, call.owner);
        if (!entry || entry->buffers.size() != call.buffers.size() ||
            !std::equal(call.buffers.begin(), call.buffers.end(), entry->buffers.begin()))
            return false;
        entry->graph.run(stream_);
        return true;
    }
    template <class Forward>
    bool run(const char *name, std::initializer_list<int> shape, bool enabled,
             Forward &&forward) const {
        ready();
        if (auto *graph = enabled ? find(name, shape) : nullptr) {
            graph->graph.run(stream_);
            return true;
        }
        forward();
        return false;
    }
};
} // namespace tts
