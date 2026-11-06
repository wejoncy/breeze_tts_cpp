#pragma once
#include <cuda_runtime_api.h>
#include <functional>
#include <initializer_list>
#include <memory>
#include <vector>

namespace tts {
struct ForwardCall;
// One graph registry for every model. Capture allocates only during initialization;
// sealed execution borrows stable addresses and performs no registry mutation.
class CUDAGraphRunner {
    struct Impl;
    std::unique_ptr<Impl> impl_;
    bool replay(const char *name, std::initializer_list<int> shape, bool enabled) const;

  public:
    using Shape = std::vector<int>;
    explicit CUDAGraphRunner(cudaStream_t stream);
    ~CUDAGraphRunner();
    CUDAGraphRunner(const CUDAGraphRunner &) = delete;
    CUDAGraphRunner &operator=(const CUDAGraphRunner &) = delete;
    bool sealed() const;
    void clear();
    void register_shape(const char *name, std::initializer_list<int> shape);
    void capture(const char *name, const std::function<void(const Shape &)> &forward);
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
