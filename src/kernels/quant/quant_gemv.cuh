#pragma once
#include <cuda_bf16.h>
#include "kernels/quant/quant_format.cuh"
// Small-M FP8 GEMV with CTA-local split K and no intermediate layout conversion.
template <int Tokens, int Splits = 1, tts::Type Storage = tts::Type::FP8>
__global__ void quant_gemv(const unsigned char *x, const float *sx, const unsigned char *w,
                           const float *sw, __nv_bfloat16 *y, int rows, int n, int k) {
    using F = tts::quant_device::Format<Storage>;
    using Acc = typename F::Acc;
    extern __shared__ unsigned int shared_x[];
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32, row = blockIdx.y * Tokens;
    int col = blockIdx.x * (8 / Splits) + warp / Splits;
    int split = warp % Splits;
    __shared__ Acc partial[8][Tokens];
    int words = k / 4, valid_words = min(Tokens, rows - row) * words;
    for (int i = threadIdx.x; i < Tokens * k / 4; i += blockDim.x) {
        shared_x[i] = i < valid_words
                          ? reinterpret_cast<const unsigned int *>(x)[size_t(row) * words + i]
                          : 0;
    }
    __syncthreads();
    Acc sums[Tokens] = {};
    if (col < n) {
#pragma unroll 4
        for (int c = lane * 4 + split * 128; c < k; c += 128 * Splits) {
            unsigned int b = *reinterpret_cast<const unsigned int *>(w + size_t(col) * k + c);
#pragma unroll
            for (int r = 0; r < Tokens; ++r)
                sums[r] = F::dot(shared_x[(r * k + c) / 4], b, sums[r]);
        }
    }
#pragma unroll
    for (int r = 0; r < Tokens; ++r) {
        for (int d = 16; d; d /= 2)
            sums[r] += __shfl_down_sync(0xffffffff, sums[r], d);
        if (!lane)
            partial[warp][r] = sums[r];
    }
    __syncthreads();
    if (!lane && !split && col < n) {
#pragma unroll
        for (int r = 0; r < Tokens; ++r) {
            Acc v = 0;
#pragma unroll
            for (int s = 0; s < Splits; ++s)
                v += partial[warp + s][r];
            if (row + r < rows)
                y[size_t(row + r) * n + col] = __nv_bfloat16(v * sw[col] * sx[row + r]);
        }
    }
}
