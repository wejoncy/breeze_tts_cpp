#include "ops/cuda/execution_context.h"
#include "runtime/cuda_context.h"

namespace tts {
namespace {
thread_local CudaContext *active_context = nullptr;
}
ExecutionScope::ExecutionScope(CudaContext &cuda) : previous_(active_context) { active_context = &cuda; }
ExecutionScope::~ExecutionScope() { active_context = previous_; }
void initialize_execution(CudaContext &cuda) {
    CUDA(cudaSetDevice(cuda.device));
    active_context = &cuda;
}
CudaContext &execution_context() {
    if (!active_context) throw std::invalid_argument("tensor allocation has no active device context");
    return *active_context;
}
CudaContext &execution_context(int device) {
    if (!active_context || device < 0 || active_context->device != device)
        throw std::invalid_argument("tensor device has no active execution context");
    return *active_context;
}
} // namespace tts
