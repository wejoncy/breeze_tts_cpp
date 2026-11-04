#include "ops/cuda/backend.h"
#include "ops/kernels/attention/relative_attention.h"

namespace tts::ops {
void relative_attention(TensorView qkv, TensorView bias, TensorView keys,
                        TensorView values, TensorView output, TensorView scratch,
                        int batch, int time, int heads, int window, const SequenceLayout *packed) {
    auto &gpu = execution_context(qkv, {bias, keys, values, output, scratch});
    if (!scratch.dense())
        throw std::invalid_argument("relative attention requires dense scratch storage");
    cuda_relative_attention(gpu.blas, gpu.stream, qkv, bias, keys, values, output, scratch.data,
                            scratch.numel() * bytes(scratch.type), batch, time, heads, window,
                            gpu.shared_memory_per_block(), packed);
}
} // namespace tts::ops
