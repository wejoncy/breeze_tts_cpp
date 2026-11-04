#pragma once
#include "core/tensor.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace tts::cuda_kernels {
template <class Launch> void dispatch_storage(Type type, Launch &&launch) {
    switch (type) {
    case Type::BF16: launch(__nv_bfloat16{}); break;
    case Type::F16: launch(__half{}); break;
    case Type::F32: launch(float{}); break;
    default: throw std::invalid_argument("operator requires F16, BF16 or F32 storage");
    }
}
} // namespace tts::cuda_kernels
