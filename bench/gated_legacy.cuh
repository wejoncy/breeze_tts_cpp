#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cfloat>
// Historical kernel for reproducible before/after benchmarks only.
namespace tts::bench {
using bf16 = __nv_bfloat16;
template <int Threads = 256> __device__ float reduce_max(float value) {
    __shared__ float partial[Threads / 32];
    for (int d = 16; d; d /= 2)
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, d));
    if (!(threadIdx.x % 32))
        partial[threadIdx.x / 32] = value;
    __syncthreads();
    int lane = threadIdx.x % 32;
    float result = lane < Threads / 32 ? partial[lane] : 0.f;
    for (int d = 16; d; d /= 2)
        result = fmaxf(result, __shfl_down_sync(0xffffffff, result, d));
    return __shfl_sync(0xffffffff, result, 0);
}
template <int Kind, int Threads = 1024>
__global__ void legacy_gated_quantize(const bf16 *input, unsigned char *out, float *scales,
                                      int cols) {
    int row = blockIdx.x;
    float values[8192 / Threads], amax = 0;
#pragma unroll
    for (int j = 0; j < 8192 / (Threads * 4); ++j) {
        int c = threadIdx.x * 4 + j * Threads * 4;
        uint2 gate{}, up{};
        if (c < cols) {
            gate = *reinterpret_cast<const uint2 *>(input + size_t(row) * cols * 2 + c);
            up = *reinterpret_cast<const uint2 *>(input + size_t(row) * cols * 2 + cols + c);
        }
        const bf16 *g = reinterpret_cast<const bf16 *>(&gate);
        const bf16 *u = reinterpret_cast<const bf16 *>(&up);
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            float x = float(g[v]);
            float activated =
                Kind == 0
                    ? x / (1.f + expf(-x))
                    : .5f * x * (1.f + tanhf(.7978845608028654f * (x + .044715f * x * x * x)));
            float value = float(bf16(float(bf16(activated)) * float(u[v])));
            values[j * 4 + v] = value;
            amax = fmaxf(amax, fabsf(value));
        }
    }
    amax = reduce_max<Threads>(amax);
    float scale = (amax == 0 ? 1.f : fmaxf(amax / 448.f, FLT_MIN)), inv = 1.f / scale;
    if (!threadIdx.x)
        scales[row] = scale;
#pragma unroll
    for (int j = 0; j < 8192 / (Threads * 4); ++j) {
        int c = threadIdx.x * 4 + j * Threads * 4;
        if (c < cols) {
            float4 v = make_float4(values[j * 4] * inv, values[j * 4 + 1] * inv,
                                   values[j * 4 + 2] * inv, values[j * 4 + 3] * inv);
            *reinterpret_cast<unsigned int *>(out + size_t(row) * cols + c) =
                __nv_fp8x4_e4m3(v).__x;
        }
    }
}
} // namespace tts::bench
