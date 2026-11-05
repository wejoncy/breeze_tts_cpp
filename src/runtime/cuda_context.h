#pragma once
#include "runtime/device.h"
#include "ops/cuda/execution_context.h"
#include "ops/cuda/quantization.h"
#include <array>
#include <map>
#include <cublasLt.h>

namespace tts {
class LtLinear;
class QuantizedGemm;
// One NVIDIA execution resource set. Operators are free functions in ops.
struct CudaContext {
    std::shared_ptr<DeviceContext> memory_owner;
    DeviceContext &memory;
    const int device;
    const cudaStream_t stream;
    Buffer gemm_workspace, attention_lse, decode_output, decode_lse, fft_workspace;
    std::map<std::pair<int, int>, int> fft_plans; // Opaque cuFFT plan identifiers.
    std::array<int, 4> gemv_slots{};
    std::array<int, 8> flash_decode_slots{};
    int qkv_gemv_capacity = 0;
    int multiprocessors = 0, sm_major = 0, sm_minor = 0;
    int normalized_linear_capacity = 0, normalized_qkv_capacity = 0;
    size_t shared_memory_per_block = 0;
    bool pdl = false, plans_sealed = false;
    cublasHandle_t blas = nullptr;
    cublasLtHandle_t lt = nullptr;
    std::unique_ptr<LtLinear> matmul;
    std::unique_ptr<QuantizedGemm> quantization;
    CudaContext(int device, Quantization, size_t memory_limit_bytes = 0);
    ~CudaContext();
    CudaContext(const CudaContext &) = delete;
};
void initialize_cuda_kernels(CudaContext &);
} // namespace tts
