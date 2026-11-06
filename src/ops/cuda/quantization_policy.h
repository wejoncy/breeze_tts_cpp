#pragma once
#include "ops/kernels/quant/kernel_config.h"
#include "core/tensor.h"
#include <string>
#include <vector>

namespace tts {
// CPU-only planning inputs. No CUDA execution, allocator or model identity.
struct ChipModel {
    int sms = 1;
    double memory_bytes_per_us = 1;
    double tensor_ops_per_us = 1;
    double scalar_ops_per_us = 1;
    double kernel_us = 1; // Graph kernel dispatch latency estimate, independent of shape.
    double warp_tensor_ops_per_us = 0;
    double warp_instructions_per_us = 1;
    size_t cache_bytes = 0;
    double cache_bytes_per_us = 0;
};
struct KernelLimits {
    bool hopper = false;
    size_t shared_bytes = 0, workspace_bytes = 0;
};
struct LinearFinishWork {
    double instructions_per_output = 0;
};
std::vector<LinearPath> quant_linear_candidates(KernelProblem, KernelLimits,
                                                Type format = Type::FP8);
std::string path_name(const LinearPath &);
double linear_pipeline_cost(KernelProblem, LinearPath, Type, const ChipModel &, LinearFinishWork,
                            bool fused, LibraryWork = {}, KernelResources = {});
LinearPath select_linear(KernelProblem, KernelLimits, Type, const ChipModel &,
                         bool library_compatible = true, LibraryWork = {});
} // namespace tts
