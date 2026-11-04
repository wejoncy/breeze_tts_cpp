#include "ops/cuda/execution_context.h"
#include "ops/cuda/backend.h"

namespace tts {
namespace {
thread_local Backend *active_context = nullptr;
}
ExecutionScope::ExecutionScope(Backend &backend) : previous_(active_context) { active_context = &backend; }
ExecutionScope::~ExecutionScope() { active_context = previous_; }
void initialize_execution(Backend &backend) {
    CUDA(cudaSetDevice(backend.device));
    active_context = &backend;
}
Backend &execution_context(int device) {
    if (!active_context || device < 0 || active_context->device != device)
        throw std::invalid_argument("tensor device has no active execution context");
    return *active_context;
}
} // namespace tts
