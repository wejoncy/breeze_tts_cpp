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
// Reads this worker's bound resources and checks the tensor device; no registry lookup.
Backend &execution_context(int device);
void initialize_execution(Backend &);
inline Backend &execution_context(TensorView tensor) { return execution_context(tensor.device); }
inline Backend &execution_context(TensorView tensor, std::initializer_list<TensorView> operands) {
    auto &backend = execution_context(tensor);
    for (auto operand : operands)
        if (operand.data && operand.device != tensor.device)
            throw std::invalid_argument("operator tensors are on different devices");
    return backend;
}
} // namespace tts
