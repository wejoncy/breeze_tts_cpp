#include "kernels/quant/quant_dispatch.h"
#include "kernels/quant/quant_gemm.h"
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
#include <cuda_bf16.h>
#include "kernels/quant/quant_format.cuh"
#include <cfloat>
namespace tts {
// BF16-input projection. The CTA computes true per-token amax and keeps the
// quantized input in shared memory. Output tiles deliberately repeat this work;
// the cost model compares it to quantize + the prepared GEMM.
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
#include <stdexcept>
namespace tts {
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
template <Type Storage>
void projection_format(int token_tile, KernelProblem shape, const void *input, const void *weights,
                       const float *sw, void *output, cudaStream_t stream) {
    int rows = shape.rows, n = shape.channels, k = shape.reduction;
    if (k != 1024 && k != 2048)
        throw std::invalid_argument("unsupported fused projection width");
    const auto *x = static_cast<const bf16 *>(input);
    const auto *w = static_cast<const unsigned char *>(weights);
    auto *y = static_cast<bf16 *>(output);
#define PROJECT(K, T)                                                                              \
    quant_projection_token_tile<K, T, Storage>                                                     \
        <<<dim3((n + 7) / 8, (rows + T - 1) / T), 256, 0, stream>>>(x, w, sw, y, rows, n)
#define PROJECT_WIDTH(T)                                                                           \
    if (k == 1024) {                                                                               \
        PROJECT(1024, T);                                                                          \
    } else {                                                                                       \
        PROJECT(2048, T);                                                                          \
    }
    switch (token_tile) {
    case 1:
        PROJECT_WIDTH(1);
        break;
    case 2:
        PROJECT_WIDTH(2);
        break;
    case 4:
        PROJECT_WIDTH(4);
        break;
    case 8:
        PROJECT_WIDTH(8);
        break;
    default:
        throw std::logic_error("invalid prepared projection tile");
    }
#undef PROJECT_WIDTH
#undef PROJECT
}
void launch_quant_linear(LinearPath config, KernelProblem shape, const void *input, const float *xs,
                         const void *weights, const float *ws, void *y, void *workspace,
                         size_t bytes, cudaStream_t stream, Type format) {
    if (format == Type::I8)
        linear_format<Type::I8>(config, shape, input, xs, weights, ws, y, workspace, bytes, stream);
    else
        linear_format<Type::FP8>(config, shape, input, xs, weights, ws, y, workspace, bytes,
                                 stream);
}
void launch_quant_projection(int tile, KernelProblem shape, const void *x, const void *w,
                             const float *sw, void *y, cudaStream_t stream, Type format) {
    if (format == Type::I8)
        projection_format<Type::I8>(tile, shape, x, w, sw, y, stream);
    else
        projection_format<Type::FP8>(tile, shape, x, w, sw, y, stream);
}
} // namespace tts
