#pragma once
#include "core/tensor.h"
#include <initializer_list>

namespace tts {
class CudaContext;
// A runner binds its execution resources at the thread boundary. Tensor device
// metadata drives dispatch; streams and library handles remain execution state.
class ExecutionScope {
    CudaContext *previous_;

  public:
    explicit ExecutionScope(CudaContext &);
    ~ExecutionScope();
    ExecutionScope(const ExecutionScope &) = delete;
    ExecutionScope &operator=(const ExecutionScope &) = delete;
};
// Reads this worker's bound resources and checks the tensor device; no registry lookup.
CudaContext &execution_context(int device);
CudaContext &execution_context();
void initialize_execution(CudaContext &);
inline CudaContext &execution_context(TensorView tensor) { return execution_context(tensor.device); }
inline CudaContext &execution_context(TensorView tensor, std::initializer_list<TensorView> operands) {
    auto &cuda = execution_context(tensor);
    for (auto operand : operands)
        if (operand.data && operand.device != tensor.device)
            throw std::invalid_argument("operator tensors are on different devices");
    return cuda;
}
} // namespace tts
