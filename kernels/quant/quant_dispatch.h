#pragma once
#include "kernels/quant/kernel_config.h"
#include "core/tensor.h"
#include <cuda_runtime.h>
namespace tts {
void launch_quant_linear(LinearPath config, KernelProblem shape, const void *input,
                         const float *input_scales, const void *weights, const float *weight_scales,
                         void *output, void *workspace, size_t workspace_bytes, cudaStream_t stream,
                         Type format = Type::FP8);
} // namespace tts
