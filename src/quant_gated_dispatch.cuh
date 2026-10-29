#pragma once
#include "quant_gated_config.h"
#include "quant_gated_kernels.cuh"
#include "device.h"
namespace tts::gated_dispatch {
namespace {
using bf16 = __nv_bfloat16;
// Keep host launch wrappers local to the CUDA module they were compiled with.
// Otherwise another executable/DSO can interpose a different-architecture image.
inline bool supports_clusters(int device) {
    int supported = 0;
    CUDA(cudaDeviceGetAttribute(&supported, cudaDevAttrClusterLaunch, device));
    if (!supported)
        return false;
    cudaFuncAttributes compiled{};
    CUDA(cudaFuncGetAttributes(
        &compiled, gated_kernels::vector_gated_quantize<0, 128, 4, false, 2, Type::I8>));
    // PTX compiled below SM90 omits the cross-CTA reduction, even when JITed on Hopper.
    return compiled.ptxVersion >= 90;
}
template <int Kind, int T, int V, int Parts, bool Table, Type Storage>
void run(const bf16 *x, unsigned char *q, float *s, int rows, int width, const bf16 *table,
         cudaStream_t stream) {
    if constexpr (Parts == 1)
        gated_kernels::vector_gated_quantize<Kind, T, V, Table, 1, Storage>
            <<<rows, T, 0, stream>>>(x, q, s, width, table);
    else {
        cudaLaunchConfig_t cfg{};
        cfg.gridDim = dim3(rows * Parts);
        cfg.blockDim = dim3(T);
        cfg.stream = stream;
        cudaLaunchAttribute attr{};
        attr.id = cudaLaunchAttributeClusterDimension;
        attr.val.clusterDim = {Parts, 1, 1};
        cfg.attrs = &attr;
        cfg.numAttrs = 1;
        CUDA(cudaLaunchKernelEx(
            &cfg, gated_kernels::vector_gated_quantize<Kind, T, V, Table, Parts, Storage>, x, q, s,
            width, table));
    }
}
template <int Kind, int T, int V, int Parts, Type Storage>
void table_run(bool lut, const bf16 *x, unsigned char *q, float *s, int rows, int width,
               const bf16 *table, cudaStream_t stream) {
    if (lut)
        run<Kind, T, V, Parts, true, Storage>(x, q, s, rows, width, table, stream);
    else
        run<Kind, T, V, Parts, false, Storage>(x, q, s, rows, width, table, stream);
}
template <int Kind, Type Storage = Type::FP8>
void launch(GatedConfig c, const bf16 *x, unsigned char *q, float *s, int rows, int width,
            const bf16 *table, cudaStream_t stream) {
    if (c.vector == 1) {
        gated_kernels::scalar_gated_quantize<Kind, Storage>
            <<<rows, 256, 0, stream>>>(x, q, s, width);
        return;
    }
#define RUN(T, V, P) table_run<Kind, T, V, P, Storage>(c.table, x, q, s, rows, width, table, stream)
#define VECTORS(T)                                                                                 \
    case T:                                                                                        \
        switch (c.vector) {                                                                        \
        case 2:                                                                                    \
            RUN(T, 2, 1);                                                                          \
            break;                                                                                 \
        case 4:                                                                                    \
            RUN(T, 4, 1);                                                                          \
            break;                                                                                 \
        case 8:                                                                                    \
            RUN(T, 8, 1);                                                                          \
            break;                                                                                 \
        default:                                                                                   \
            throw std::logic_error("invalid gated vector");                                        \
        }                                                                                          \
        break
    if (c.clusters == 1) {
        switch (c.threads) {
            VECTORS(128);
            VECTORS(256);
            VECTORS(512);
            VECTORS(1024);
        default:
            throw std::logic_error("invalid gated block");
        }
        return;
    }
#undef VECTORS
#define CLUSTER(P)                                                                                 \
    case P:                                                                                        \
        if (c.threads == 128) {                                                                    \
            if (c.vector == 4) {                                                                   \
                RUN(128, 4, P);                                                                    \
            } else {                                                                               \
                RUN(128, 8, P);                                                                    \
            }                                                                                      \
        } else {                                                                                   \
            if (c.vector == 4) {                                                                   \
                RUN(256, 4, P);                                                                    \
            } else {                                                                               \
                RUN(256, 8, P);                                                                    \
            }                                                                                      \
        }                                                                                          \
        break
    switch (c.clusters) {
        CLUSTER(2);
        CLUSTER(4);
        CLUSTER(8);
    default:
        throw std::logic_error("invalid gated cluster");
    }
#undef CLUSTER
#undef RUN
}
template <Type Storage = Type::FP8>
inline void launch(GatedConfig c, int kind, const bf16 *x, unsigned char *q, float *s, int rows,
                   int width, const bf16 *table, cudaStream_t stream) {
    if (kind == 0)
        launch<0, Storage>(c, x, q, s, rows, width, table, stream);
    else
        launch<1, Storage>(c, x, q, s, rows, width, table, stream);
}
inline void launch(Type format, GatedConfig c, int kind, const bf16 *x, unsigned char *q, float *s,
                   int rows, int width, const bf16 *table, cudaStream_t stream) {
    if (format == Type::I8)
        launch<Type::I8>(c, kind, x, q, s, rows, width, table, stream);
    else
        launch<Type::FP8>(c, kind, x, q, s, rows, width, table, stream);
}
} // namespace
} // namespace tts::gated_dispatch
