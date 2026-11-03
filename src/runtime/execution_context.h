#pragma once
#include "core/tensor.h"
#include <initializer_list>

namespace tts {
class Backend;
// A runner binds its execution resources at the thread boundary. Tensor device
// metadata drives dispatch; streams and library handles remain execution state.
class ExecutionScope {
    Backend *previous_;

  public:
    explicit ExecutionScope(Backend &);
    ~ExecutionScope();
    ExecutionScope(const ExecutionScope &) = delete;
    ExecutionScope &operator=(const ExecutionScope &) = delete;
};
Backend &backend_for(int device);
inline Backend &backend_for(TensorView tensor) { return backend_for(tensor.device); }
inline Backend &backend_for(TensorView tensor, std::initializer_list<TensorView> operands) {
    auto &backend = backend_for(tensor);
    for (auto operand : operands)
        if (operand.data && operand.device != tensor.device)
            throw std::invalid_argument("operator tensors are on different devices");
    return backend;
}
} // namespace tts
