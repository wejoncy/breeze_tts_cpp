#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "core/tensor.h"
namespace tts {
// Split-K partials remain in the accumulator type until the final scaled store.
template <class Acc, class D>
static __global__ void quant_split_reduce_scale(const Acc *partial, const float *sx,
                                                const float *sw, D *y, int m, int n, int splits) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < m * n) {
        Acc v = 0;
        for (int s = 0; s < splits; ++s)
            v += partial[size_t(s) * m * n + i];
        y[i] = D(v * sx[i / n] * sw[i % n]);
    }
}
template <class Acc>
static void launch_quant_split_reduce(const Acc *partial, const float *sx, const float *sw, void *y, int m,
                               int n, int splits, cudaStream_t stream, Type output_type) {
    if (output_type == Type::F16)
        quant_split_reduce_scale<<<(m * n + 255) / 256, 256, 0, stream>>>(
            partial, sx, sw, static_cast<__half *>(y), m, n, splits);
    else
        quant_split_reduce_scale<<<(m * n + 255) / 256, 256, 0, stream>>>(
            partial, sx, sw, static_cast<__nv_bfloat16 *>(y), m, n, splits);
}
} // namespace tts
