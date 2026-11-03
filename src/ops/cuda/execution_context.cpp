#include "runtime/execution_context.h"
#include "ops/backend.h"

namespace tts {
namespace {
thread_local Backend *current = nullptr;
}
ExecutionScope::ExecutionScope(Backend &backend) : previous_(current) { current = &backend; }
ExecutionScope::~ExecutionScope() { current = previous_; }
Backend &backend_for(int device) {
    if (!current || device < 0 || current->device != device)
        throw std::invalid_argument("tensor device has no active execution context");
    return *current;
}
} // namespace tts
