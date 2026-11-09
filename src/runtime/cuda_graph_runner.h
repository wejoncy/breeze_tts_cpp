#pragma once
#include <cuda_runtime_api.h>
#include "runtime/graph.h"
#include <initializer_list>
#include <memory>
#include <vector>

namespace tts {
struct ForwardCall;
// One graph registry for every model. Capture allocates only during initialization;
// sealed execution borrows stable addresses and performs no registry mutation.
class CUDAGraphRunner {
  public:
    using Shape = std::vector<int>;
  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    cudaStream_t stream_;
    std::vector<Shape> capture_shapes(const char *);
    Graph &capture_target(const char *, const Shape &);
    bool replay(const char *name, std::initializer_list<int> shape, bool enabled) const;

  public:
    explicit CUDAGraphRunner(cudaStream_t stream);
    ~CUDAGraphRunner();
    CUDAGraphRunner(const CUDAGraphRunner &) = delete;
    CUDAGraphRunner &operator=(const CUDAGraphRunner &) = delete;
    bool sealed() const;
    void clear();
    void register_shape(const char *name, std::initializer_list<int> shape);
    template <class Forward> void capture(const char *name, Forward &&forward) {
        const auto shapes = capture_shapes(name);
        GraphCaptureProgress progress(name, shapes.size());
        for (const auto &shape : shapes) {
            capture_target(name, shape).capture(stream_, [&] { forward(shape); });
            progress.advance();
        }
    }
    void seal();
    bool capture(const ForwardCall &call);
    bool run(const ForwardCall &call) const;
    template <class Forward>
    bool run(const char *name, std::initializer_list<int> shape, bool enabled,
             Forward &&forward) const {
        if (replay(name, shape, enabled))
            return true;
        forward();
        return false;
    }
};
} // namespace tts
