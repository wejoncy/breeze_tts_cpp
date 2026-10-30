#include "kernels/quant/quant_dispatch.h"
#include "kernels/quant/quant_gemm.h"
#include <cuda_bf16.h>
#include "kernels/quant/quant_format.cuh"
#include <stdexcept>
namespace tts {
namespace {
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
using bf16 = __nv_bfloat16;
template <Type Storage>
void linear_format(LinearPath config, KernelProblem shape, const void *input, const float *xs,
                   const void *weights, const float *ws, void *y, void *workspace,
                   size_t workspace_bytes, cudaStream_t stream) {
    int m = shape.rows, n = shape.channels, k = shape.reduction;
    auto *x = const_cast<void *>(input), *w = const_cast<void *>(weights);
#define GEMV(T, S)                                                                                 \
    quant_gemv<T, S, Storage>                                                                      \
        <<<dim3((n + (8 / S) - 1) / (8 / S), (m + T - 1) / T), 256, T * k, stream>>>(              \
            static_cast<unsigned char *>(x), xs, static_cast<unsigned char *>(w), ws,              \
            static_cast<bf16 *>(y), m, n, k)
    if (config.kind == LinearKind::Gemv) {
        if (config.tokens == 2) {
            GEMV(2, 1);
        } else if (config.tokens == 4) {
            GEMV(4, 1);
        } else if (config.splits == 2) {
            GEMV(1, 2);
        } else if (config.splits == 4) {
            GEMV(1, 4);
        } else {
            GEMV(1, 1);
        }
        return;
    }
    if constexpr (Storage == Type::I8) {
        if (config.kind == LinearKind::Sm80) {
            int8_sm80_gemm(x, xs, w, ws, y, m, n, k, workspace, workspace_bytes, stream,
                           config.channels, config.tokens, config.splits);
            return;
        }
    }
    if (config.kind == LinearKind::Hopper) {
        auto gemm = Storage == Type::I8 ? int8_hopper_gemm : fp8_hopper_gemm;
        gemm(x, xs, w, ws, y, m, n, k, workspace, workspace_bytes, stream, config.channels,
             config.tokens, config.splits);
        return;
    }
#undef GEMV
    throw std::invalid_argument("library GEMM must use its prepared descriptor");
}
} // namespace
void launch_quant_linear(LinearPath config, KernelProblem shape, const void *input, const float *xs,
                         const void *weights, const float *ws, void *y, void *workspace,
                         size_t bytes, cudaStream_t stream, Type format) {
    if (format == Type::I8)
        linear_format<Type::I8>(config, shape, input, xs, weights, ws, y, workspace, bytes, stream);
    else
        linear_format<Type::FP8>(config, shape, input, xs, weights, ws, y, workspace, bytes,
                                 stream);
}
} // namespace tts
