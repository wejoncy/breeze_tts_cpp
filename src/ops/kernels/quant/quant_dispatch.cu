#include "ops/kernels/quant/quant_dispatch.h"
#include "ops/kernels/quant/quant_gemm.h"
#include "ops/sequence.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "ops/kernels/quant/quant_format.cuh"
#include <stdexcept>
#include <type_traits>
namespace tts {
namespace {
// Small-M FP8 GEMV with CTA-local split K and no intermediate layout conversion.
template <int Tokens, int Splits, tts::Type Storage, class D, bool Grouped>
__global__ void quant_gemv(const unsigned char *x, const float *sx, const unsigned char *w,
                           const float *sw, D *y, int rows, int n, int k, int input_group,
                           int weight_group) {
    using F = tts::quant_device::Format<Storage>;
    using Acc = std::conditional_t<Grouped, float, typename F::Acc>;
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
    float *shared_sx = reinterpret_cast<float *>(shared_x + Tokens * words);
    int input_groups = input_group ? (k + input_group - 1) / input_group : 0;
    int weight_groups = weight_group ? (k + weight_group - 1) / weight_group : 0;
    float *shared_sw = shared_sx + Tokens * input_groups;
    if constexpr (Grouped) {
        for (int index = threadIdx.x; index < Tokens * input_groups; index += blockDim.x) {
            int token = index % Tokens, group = index / Tokens;
            shared_sx[index] = row + token < rows ? sx[size_t(group) * rows + row + token] : 0.f;
        }
        constexpr int columns = 8 / Splits;
        for (int index = threadIdx.x; index < columns * weight_groups; index += blockDim.x) {
            int column = blockIdx.x * columns + index % columns, group = index / columns;
            shared_sw[index] = column < n ? sw[size_t(group) * n + column] : 0.f;
        }
    }
    __syncthreads();
    Acc sums[Tokens] = {};
    int input_shift = __ffs(input_group) - 1, weight_shift = __ffs(weight_group) - 1;
    auto accumulate = [&](int token, int word, unsigned packed) {
        if constexpr (Grouped) {
            unsigned value = word < words ? shared_x[token * words + word] : 0;
            auto partial = F::dot(value, packed, 0);
            if (row + token < rows && word < words) {
                float scale = input_group
                                  ? shared_sx[size_t((word * 4) >> input_shift) * Tokens + token]
                                  : 1.f;
                if (weight_group)
                    scale *= shared_sw[size_t((word * 4) >> weight_shift) * (8 / Splits) +
                                       warp / Splits];
                sums[token] += float(partial) * scale;
            }
        } else {
            sums[token] = F::dot(shared_x[token * words + word], packed, sums[token]);
        }
    };
    if (col < n) {
        const auto *weight = reinterpret_cast<const unsigned int *>(w) + size_t(col) * words;
        int word = lane + split * 32;
        // Stage the existing four-way unroll before consuming its packed weights.
#pragma unroll 1
        for (; (Grouped ? word - lane + 3 * 32 * Splits + 31 : word + 3 * 32 * Splits) < words;
             word += 4 * 32 * Splits) {
            unsigned int packed[4];
#pragma unroll
            for (int i = 0; i < 4; ++i)
                packed[i] = weight[word + i * 32 * Splits];
#pragma unroll
            for (int r = 0; r < Tokens; ++r) {
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    accumulate(r, word + i * 32 * Splits, packed[i]);
            }
        }
#pragma unroll 1
        for (; (Grouped ? word - lane : word) < words; word += 32 * Splits) {
            unsigned int b;
            if constexpr (Grouped)
                b = word < words ? weight[word] : 0;
            else
                b = weight[word];
#pragma unroll
            for (int r = 0; r < Tokens; ++r)
                accumulate(r, word, b);
        }
    }
#pragma unroll
    for (int r = 0; r < Tokens; ++r) {
        if constexpr (Storage == Type::I8 && !Grouped)
            sums[r] = __reduce_add_sync(0xffffffffu, sums[r]);
        else
            for (int d = 16; d; d /= 2)
                sums[r] += __shfl_down_sync(0xffffffff, sums[r], d);
        if constexpr (Splits > 1)
            if (!lane)
                partial[warp][r] = sums[r];
    }
    if constexpr (Splits > 1)
        __syncthreads();
    if (!lane && !split && col < n) {
#pragma unroll
        for (int r = 0; r < Tokens; ++r) {
            Acc v = 0;
            if constexpr (Splits == 1)
                v += sums[r];
            else {
#pragma unroll
                for (int s = 0; s < Splits; ++s)
                    v += partial[warp + s][r];
            }
            if (row + r < rows) {
                if constexpr (Grouped) {
                    float scale = weight_group ? 1.f : sw[col];
                    if (!input_group)
                        scale *= sx[row + r];
                    y[size_t(row + r) * n + col] = D(v * scale);
                } else {
                    y[size_t(row + r) * n + col] = D(v * sw[col] * sx[row + r]);
                }
            }
        }
    }
}
using bf16 = __nv_bfloat16;
template <Type Storage, class D, bool Grouped>
void linear_format(LinearPath config, KernelProblem shape, const void *input, const float *xs,
                   const void *weights, const float *ws, void *y, void *workspace,
                   size_t workspace_bytes, cudaStream_t stream, Type output_type) {
    int m = shape.rows, n = shape.channels, k = shape.reduction;
    auto *x = const_cast<void *>(input), *w = const_cast<void *>(weights);
#define GEMV(T, S)                                                                                 \
    quant_gemv<T, S, Storage, D, Grouped>                                                          \
        <<<dim3((n + (8 / S) - 1) / (8 / S), (m + T - 1) / T), 256,                                \
           gemv_shared_bytes(shape, T, 8 / S) - 8 * T * sizeof(float), stream>>>(                  \
            static_cast<unsigned char *>(x), xs, static_cast<unsigned char *>(w), ws,              \
            static_cast<D *>(y), m, n, k, shape.input_group, shape.weight_group)
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
    if constexpr (Grouped) {
#if TTS_HAVE_HOPPER_KERNELS
        if constexpr (Storage == Type::FP8)
            if (config.kind == LinearKind::Hopper) {
                fp8_hopper_grouped(shape, input, xs, weights, ws, y, workspace, workspace_bytes,
                                   stream, output_type, config.tokens, config.splits);
                return;
            }
#endif
        if (config.kind != LinearKind::Grouped)
            throw std::invalid_argument("group scales require a grouped GEMM implementation");
        grouped_quant_gemm(shape, input, xs, weights, ws, y, stream, Storage, output_type,
                           config.tokens, config.splits, workspace, workspace_bytes);
        return;
    }
    if constexpr (Storage == Type::I8) {
        if (config.kind == LinearKind::Sm80) {
            int8_sm80_gemm(x, xs, w, ws, y, m, n, k, workspace, workspace_bytes, stream,
                           config.channels, config.tokens, config.splits, output_type);
            return;
        }
    }
    if (config.kind == LinearKind::Hopper) {
#if TTS_HAVE_HOPPER_KERNELS
        auto gemm = Storage == Type::I8 ? int8_hopper_gemm : fp8_hopper_gemm;
        gemm(x, xs, w, ws, y, m, n, k, workspace, workspace_bytes, stream, config.channels,
             config.tokens, config.splits, output_type);
        return;
#else
        throw std::runtime_error("native Hopper kernels are disabled in this build");
#endif
    }
#undef GEMV
    throw std::invalid_argument("library GEMM must use its prepared descriptor");
}
} // namespace
void launch_quant_linear(LinearPath config, KernelProblem shape, const void *input, const float *xs,
                         const void *weights, const float *ws, void *y, void *workspace,
                         size_t bytes, cudaStream_t stream, Type format, Type output_type,
                         const ops::Epilogue *finish) {
    if (output_type != Type::BF16 && output_type != Type::F16)
        throw std::invalid_argument("quantized GEMM output must be F16 or BF16");
    if (finish) {
#if TTS_HAVE_HOPPER_KERNELS
        if (format == Type::I8 && config.kind == LinearKind::Hopper && config.splits == 1) {
            int8_hopper_finish(const_cast<void *>(input), xs, const_cast<void *>(weights), ws, y,
                               shape.rows, shape.channels, shape.reduction, workspace, bytes,
                               stream, config.channels, config.tokens, output_type,
                               finish->bias.data, finish->activation);
            return;
        }
#endif
        throw std::invalid_argument("unsupported quantized native Finish path");
    }
#define RUN(F, D)                                                                                  \
    if (shape.input_group || shape.weight_group)                                                   \
        linear_format<F, D, true>(config, shape, input, xs, weights, ws, y, workspace, bytes,      \
                                  stream, output_type);                                            \
    else                                                                                           \
        linear_format<F, D, false>(config, shape, input, xs, weights, ws, y, workspace, bytes,     \
                                   stream, output_type)
    if (format == Type::I8) {
        if (output_type == Type::F16) {
            RUN(Type::I8, __half);
        } else {
            RUN(Type::I8, bf16);
        }
    } else if (format == Type::FP8) {
        if (output_type == Type::F16) {
            RUN(Type::FP8, __half);
        } else {
            RUN(Type::FP8, bf16);
        }
    } else
        throw std::invalid_argument("unsupported quantized storage type");
#undef RUN
}
KernelResources quant_linear_finish_resources(LinearPath path, Type format, Type output_type) {
#if TTS_HAVE_HOPPER_KERNELS
    if (format == Type::I8 && path.kind == LinearKind::Hopper && path.splits == 1)
        return int8_hopper_finish_resources(path.channels, path.tokens, output_type);
#endif
    return {};
}
} // namespace tts
