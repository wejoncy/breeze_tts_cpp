#pragma once
#include "quant_gated_config.h"
#include "kernels/quant/quant_gated_launch.cuh"
namespace tts::gated_dispatch {
namespace {
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
        gated_kernels::vector_gated_quantize<Kind, 256, 2, false, 1, Storage>
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
