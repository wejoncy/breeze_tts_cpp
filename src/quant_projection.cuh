#pragma once
#include <cuda_bf16.h>
#include "quant_format.cuh"
#include <cfloat>
namespace tts {
// BF16-input projection. The CTA computes true per-token amax and keeps the
// quantized input in shared memory. Output tiles deliberately repeat this work;
// admission-time timing compares it to quantize + the prepared GEMM.
using bf16 = __nv_bfloat16;
template <int K, int Tokens, Type Storage = Type::FP8>
__global__ void quant_projection_token_tile(const bf16 *x, const unsigned char *w, const float *sw,
                                            bf16 *y, int rows, int n) {
    using F = quant_device::Format<Storage>;
    typename F::Acc sums[Tokens] = {};
    __shared__ unsigned int aq[Tokens * K / 4];
    __shared__ float partial[Tokens * 8], scales[Tokens], inverse[Tokens];
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32, row = blockIdx.y * Tokens,
        col = blockIdx.x * 8 + warp;
    {
        float values[Tokens][K / 256];
#pragma unroll
        for (int r = 0; r < Tokens; ++r) {
            float amax = 0;
#pragma unroll
            for (int j = 0; j < K / 256; ++j) {
                int c = threadIdx.x + j * 256;
                float value = row + r < rows ? float(x[size_t(row + r) * K + c]) : 0;
                values[r][j] = value;
                amax = fmaxf(amax, fabsf(value));
            }
            for (int d = 16; d; d /= 2)
                amax = fmaxf(amax, __shfl_down_sync(0xffffffff, amax, d));
            if (!lane)
                partial[r * 8 + warp] = amax;
        }
        __syncthreads();
        if (threadIdx.x < Tokens) {
            float amax = 0;
#pragma unroll
            for (int w = 0; w < 8; ++w)
                amax = fmaxf(amax, partial[threadIdx.x * 8 + w]);
            scales[threadIdx.x] = quant_device::scale<Storage>(amax);
            inverse[threadIdx.x] = 1.f / scales[threadIdx.x];
        }
        __syncthreads();
#pragma unroll
        for (int r = 0; r < Tokens; ++r) {
#pragma unroll
            for (int j = 0; j < K / 256; ++j)
                reinterpret_cast<unsigned char *>(aq)[r * K + threadIdx.x + j * 256] =
                    F::encode(values[r][j] * inverse[r]);
        }
        __syncthreads();
    }
    if (col < n) {
#pragma unroll 4
        for (int c = lane * 4; c < K; c += 128) {
            unsigned int b = *reinterpret_cast<const unsigned int *>(w + size_t(col) * K + c);
#pragma unroll
            for (int r = 0; r < Tokens; ++r)
                sums[r] = F::dot(aq[(r * K + c) / 4], b, sums[r]);
        }
    }
#pragma unroll
    for (int r = 0; r < Tokens; ++r) {
        for (int d = 16; d; d /= 2)
            sums[r] += __shfl_down_sync(0xffffffff, sums[r], d);
        if (!lane && col < n && row + r < rows) {
            float scale = scales[r];
            y[size_t(row + r) * n + col] = bf16(sums[r] * sw[col] * scale);
        }
    }
}
} // namespace tts
