#pragma once
#include "runtime/error.h"
#include <cuda_runtime.h>
#include <string>

namespace tts {
inline void check(cudaError_t status, const char *expression) {
    if (status != cudaSuccess)
        throw DeviceError(std::string(expression) + ": " + cudaGetErrorString(status));
}
} // namespace tts
#define CUDA(expression) ::tts::check((expression), #expression)
