#pragma once
#include "ops/kernels/quant/kernel_config.h"
#include "core/tensor.h"
#include <cuda_runtime.h>
namespace tts {
namespace ops { struct Epilogue; }
void launch_quant_linear(LinearPath config, KernelProblem shape, const void *input,
                         const float *input_scales, const void *weights, const float *weight_scales,
                         void *output, void *workspace, size_t workspace_bytes, cudaStream_t stream,
                         Type format = Type::FP8, Type output_type = Type::BF16,
                         const ops::Epilogue *finish = nullptr);
// Initialization only; unsupported native finish paths return empty resources.
KernelResources quant_linear_finish_resources(LinearPath, Type format, Type output_type);
} // namespace tts
