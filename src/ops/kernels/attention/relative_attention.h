#pragma once
#include "core/tensor.h"
#include "ops/attention.h"
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
namespace tts {
// Consumes the projected QKV scratch in place: bias addition and query scaling
// preserve their separate storage-precision rounding. Relative tables have shape
// [heads_or_one * (2*window+1), head_dim]. No layout conversion or allocation.
// F16/BF16 FlashAttention requires 16-byte aligned QKV rows and workspace.
void cuda_relative_attention(cublasHandle_t, cudaStream_t, TensorView qkv, TensorView bias,
                        TensorView relative_keys, TensorView relative_values, TensorView out,
                        void *workspace, size_t workspace_bytes, int batch, int time, int heads,
                        int window, size_t shared_memory_limit, const SequenceLayout *packed = nullptr);
} // namespace tts
