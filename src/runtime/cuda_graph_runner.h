#pragma once
#include "runtime/graph.h"
#include <algorithm>
#include <initializer_list>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace tts {
// Models describe graph programs with integer shape axes and supply forward.
// Registration/capture are init-only; replay only reads the sealed registry.
class CUDAGraphRunner {
  public:
    using Shape = std::vector<int>;

  private:
    struct ShapeLess {
        using is_transparent = void;
        template <class Left, class Right>
        bool operator()(const Left &a, const Right &b) const {
            return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
        }
    };
    using Program = std::map<Shape, Graph, ShapeLess>;
    cudaStream_t stream_;
    std::map<std::string, Program, std::less<>> programs_;
    bool sealed_ = false;
    void initializing() const {
        if (sealed_) throw std::logic_error("CUDA graph mutation after initialization");
    }
    void ready() const {
        if (!sealed_) throw std::logic_error("CUDA graph execution before initialization");
    }
    const Graph *find(const char *name, std::initializer_list<int> shape) const {
        if (!name || !shape.size()) return nullptr;
        auto program = programs_.find(name);
        if (program == programs_.end()) return nullptr;
        auto graph = program->second.find(shape);
        return graph == program->second.end() ? nullptr : &graph->second;
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
        programs_[name].try_emplace(Shape(shape));
    }
    template <class Forward> void capture(const char *name, Forward &&forward) {
        initializing();
        if (!name || !*name)
            throw std::invalid_argument("invalid CUDA graph program");
        auto program = programs_.find(name);
        if (program == programs_.end())
            throw std::invalid_argument("CUDA graph program was not registered");
        GraphCaptureProgress progress(name, program->second.size());
        for (auto it = program->second.rbegin(); it != program->second.rend(); ++it) {
            it->second.capture(stream_, [&] { forward(it->first); });
            progress.advance();
        }
    }
    template <class Forward> void warmup(Forward &&forward) {
        initializing();
        Graph graph;
        graph.capture(stream_, std::forward<Forward>(forward));
    }
    void seal() {
        for (const auto &program : programs_)
            for (const auto &entry : program.second)
                if (!entry.second.executable)
                    throw std::logic_error("CUDA graph program has uncaptured shapes");
        sealed_ = true;
    }
    bool can_replay(const char *name, std::initializer_list<int> shape) const {
        ready();
        return find(name, shape) != nullptr;
    }
    // Componentwise capacity selection for sparse multi-axis shape plans.
    // The returned descriptor belongs to the immutable registry, never a graph handle.
    const Shape *select(const char *name, std::initializer_list<int> required) const {
        ready();
        if (!name || !required.size()) return nullptr;
        auto program = programs_.find(name);
        if (program == programs_.end()) return nullptr;
        for (auto it = program->second.lower_bound(required); it != program->second.end(); ++it)
            if (it->first.size() == required.size() &&
                std::equal(required.begin(), required.end(), it->first.begin(),
                           [](int actual, int capacity) { return actual <= capacity; }))
                return &it->first;
        return nullptr;
    }
    void replay(const char *name, std::initializer_list<int> shape) const {
        ready();
        auto *graph = find(name, shape);
        if (!graph) throw std::logic_error("CUDA graph shape was not initialized");
        graph->run(stream_);
    }
    template <class Forward>
    bool run(const char *name, std::initializer_list<int> shape, bool enabled,
             Forward &&forward) const {
        ready();
        if (auto *graph = enabled ? find(name, shape) : nullptr) {
            graph->run(stream_);
            return true;
        }
        forward();
        return false;
    }
};
} // namespace tts
