#include "runtime/cuda_graph_runner.h"
#include "runtime/graph.h"
#include "core/forward_context.h"
#include "ops/cuda/cuda_check.h"
#include "support/log.h"
#include <algorithm>
#include <map>
#include <string>
#include <utility>

namespace tts {
GraphCaptureProgress::GraphCaptureProgress(const char *name, size_t total)
    : name_(name), total_(total) {
    if (total_)
        report();
}
void GraphCaptureProgress::advance() {
    ++done_;
    const auto now = std::chrono::steady_clock::now();
    if (done_ != total_ && now < next_report_)
        return;
    report();
    next_report_ = now + std::chrono::milliseconds(200);
}
void GraphCaptureProgress::report() const {
    const size_t filled = total_ ? 20 * done_ / total_ : 20;
    TTS_LOG(TTS_LOG_INFO, "engine.init", "CUDA graphs {} [{}{}] {}/{} ({:.0f}%) elapsed_ms={:.0f}",
            name_, std::string(filled, '='), std::string(20 - filled, ' '), done_, total_,
            total_ ? 100.0 * done_ / total_ : 100.0,
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_)
                .count());
}
Graph::~Graph() {
    if (executable)
        cudaGraphExecDestroy(executable);
    if (graph)
        cudaGraphDestroy(graph);
}
void Graph::run(cudaStream_t stream) const {
    CUDA(cudaGraphLaunch(executable, stream));
}
struct CUDAGraphRunner::Impl {
  public:
    using Shape = std::vector<int>;

  private:
    struct ShapeLess {
        using is_transparent = void;
        template <class Left, class Right>
        bool operator()(const Left &left, const Right &right) const {
            return std::lexicographical_compare(left.begin(), left.end(), right.begin(),
                                                right.end());
        }
    };
    struct Entry {
        std::vector<const void *> buffers;
        Graph graph;
    };
    using Program = std::map<Shape, Entry, ShapeLess>;
    cudaStream_t stream_;
    std::map<const void *, std::map<std::string, Program, std::less<>>, std::less<const void *>>
        programs_;
    bool sealed_ = false;
    void initializing() const {
        if (sealed_)
            throw std::logic_error("CUDA graph mutation after initialization");
    }
    void ready() const {
        if (!sealed_)
            throw std::logic_error("CUDA graph execution before initialization");
    }
    const Program *program(const void *owner, const char *name) const {
        if (!name)
            return nullptr;
        auto scope = programs_.find(owner);
        if (scope == programs_.end())
            return nullptr;
        auto found = scope->second.find(name);
        return found == scope->second.end() ? nullptr : &found->second;
    }
    const Entry *find(const char *name, std::initializer_list<int> shape,
                      const void *owner = nullptr) const {
        if (!name || !shape.size())
            return nullptr;
        auto family = program(owner, name);
        if (!family)
            return nullptr;
        auto graph = family->find(shape);
        return graph == family->end() ? nullptr : &graph->second;
    }

  public:
    explicit Impl(cudaStream_t stream) : stream_(stream) {}
    Impl(const Impl &) = delete;
    Impl &operator=(const Impl &) = delete;
    bool sealed() const {
        return sealed_;
    }
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
    std::vector<Shape> capture_shapes(const char *name) {
        initializing();
        if (!name || !*name || !program(nullptr, name))
            throw std::invalid_argument("CUDA graph program was not registered");
        std::vector<Shape> shapes;
        const auto &family = programs_.at(nullptr).at(name);
        for (auto it = family.rbegin(); it != family.rend(); ++it) shapes.push_back(it->first);
        return shapes;
    }
    Graph &capture_target(const char *name, const Shape &shape) {
        initializing();
        return programs_.at(nullptr).at(name).at(shape).graph;
    }
    void seal() {
        for (const auto &owner : programs_)
            for (const auto &program : owner.second) {
                for (const auto &entry : program.second)
                    if (!entry.second.graph.executable)
                        throw std::logic_error("CUDA graph program has an uncaptured shape");
                if (owner.first)
                    TTS_LOG(TTS_LOG_DEBUG, "engine.init", "CUDA graph capture {} shapes={}",
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
    bool replay(const char *name, std::initializer_list<int> shape, bool enabled) const {
        ready();
        if (auto *graph = enabled ? find(name, shape) : nullptr) {
            graph->graph.run(stream_);
            return true;
        }
        return false;
    }
};
CUDAGraphRunner::CUDAGraphRunner(cudaStream_t stream) : impl_(std::make_unique<Impl>(stream)), stream_(stream) {}
CUDAGraphRunner::~CUDAGraphRunner() = default;
bool CUDAGraphRunner::sealed() const {
    return impl_->sealed();
}
void CUDAGraphRunner::clear() {
    impl_->clear();
}
void CUDAGraphRunner::register_shape(const char *name, std::initializer_list<int> shape) {
    impl_->register_shape(name, shape);
}
std::vector<CUDAGraphRunner::Shape> CUDAGraphRunner::capture_shapes(const char *name) {
    return impl_->capture_shapes(name);
}
Graph &CUDAGraphRunner::capture_target(const char *name, const Shape &shape) {
    return impl_->capture_target(name, shape);
}
void CUDAGraphRunner::seal() {
    impl_->seal();
}
bool CUDAGraphRunner::capture(const ForwardCall &call) {
    return impl_->capture(call);
}
bool CUDAGraphRunner::run(const ForwardCall &call) const {
    return impl_->run(call);
}
bool CUDAGraphRunner::replay(const char *name, std::initializer_list<int> shape,
                             bool enabled) const {
    return impl_->replay(name, shape, enabled);
}
} // namespace tts
