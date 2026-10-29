#pragma once
#include <cuda_bf16.h>
namespace tts {
// Split-K partials remain in the accumulator type until the final scaled store.
template <class Acc>
static __global__ void quant_split_reduce_scale(const Acc *partial, const float *sx,
                                                const float *sw, __nv_bfloat16 *y, int m, int n,
                                                int splits) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < m * n) {
        Acc v = 0;
        for (int s = 0; s < splits; ++s)
            v += partial[size_t(s) * m * n + i];
        y[i] = __nv_bfloat16(v * sx[i / n] * sw[i % n]);
    }
}
} // namespace tts
