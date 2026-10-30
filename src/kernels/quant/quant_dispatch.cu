#include "kernels/quant/quant_dispatch.h"
#include "kernels/quant/quant_gemm.h"
#include "kernels/quant/quant_gemv.cuh"
#include "kernels/quant/quant_projection.cuh"
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
