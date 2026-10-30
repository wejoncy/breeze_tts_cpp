#pragma once
#include "kernels/quant/quant_device.cuh"
#include <cuda_fp8.h>
#include <cfloat>
#include <cooperative_groups.h>
namespace tts::gated_kernels {
using bf16 = __nv_bfloat16;
template <int Kind> __device__ __forceinline__ float activate(float x) {
    return Kind == 0 ? x / (1.f + expf(-x))
                     : .5f * x * (1.f + tanhf(.7978845608028654f * (x + .044715f * x * x * x)));
}
// BF16 has 65536 possible bit patterns. Materialize the exact rounded unary
// activation once, using the same functions and operation order as the baseline.
// This table is independent of model, shape and token values.
template <int Kind> __global__ void initialize_activation_table(bf16 *table) {
    unsigned bits = blockIdx.x * blockDim.x + threadIdx.x;
    if (bits >= 65536)
        return;
    float x = float(__ushort_as_bfloat16(static_cast<unsigned short>(bits)));
    table[bits] = bf16(activate<Kind>(x));
}
// Preserve the two BF16 rounding points; retain BF16 pairs across token amax.
// Full expf/tanhf semantics match the existing producer. No fast-math substitute.
template <int Kind, int Threads, int Vec, bool Table = false, int Parts = 1,
          Type Storage = Type::FP8>
__global__ void vector_gated_quantize(const bf16 *input, unsigned char *out, float *scales,
                                      int cols, const bf16 *table = nullptr) {
    constexpr int Items = (8192 + Threads * Vec * Parts - 1) / (Threads * Vec * Parts);
    using Pack = quant_device::Bf16Pack<Vec>;
    int row = blockIdx.x / Parts;
    int shard = blockIdx.x % Parts;
    Pack values[Items];
    float amax = 0;
#pragma unroll
    for (int j = 0; j < Items; ++j) {
        int c = (threadIdx.x + (j * Parts + shard) * Threads) * Vec;
        Pack value{};
        if (c < cols) {
            Pack g = quant_device::load_pack<Pack>(input + size_t(row) * cols * 2 + c, cols - c);
            Pack u =
                quant_device::load_pack<Pack>(input + size_t(row) * cols * 2 + cols + c, cols - c);
#pragma unroll
            for (int p = 0; p < Vec / 2; ++p) {
                float2 f = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 *>(g.v + p * 2));
                __nv_bfloat162 activated;
                if constexpr (Table) {
                    unsigned short ga = __bfloat16_as_ushort(g.v[p * 2]),
                                   gb = __bfloat16_as_ushort(g.v[p * 2 + 1]);
                    activated = __halves2bfloat162(table[ga], table[gb]);
                } else {
                    activated = __floats2bfloat162_rn(activate<Kind>(f.x), activate<Kind>(f.y));
                }
                auto v = __hmul2(activated, *reinterpret_cast<__nv_bfloat162 *>(u.v + p * 2));
                *reinterpret_cast<__nv_bfloat162 *>(value.v + p * 2) = v;
                f = __bfloat1622float2(v);
                amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
            }
        }
        values[j] = value;
    }
    amax = quant_device::maximum<Threads>(amax);
    if constexpr (Parts > 1) {
#if __CUDA_ARCH__ >= 900
        __shared__ float local_max, token_max;
        if (!threadIdx.x)
            local_max = amax;
        auto cluster = cooperative_groups::this_cluster();
        cluster.sync();
        if (threadIdx.x < 32) {
            float m = threadIdx.x < Parts ? *cluster.map_shared_rank(&local_max, threadIdx.x) : 0.f;
            m = __uint_as_float(__reduce_max_sync(0xffffffffu, __float_as_uint(m)));
            if (!threadIdx.x)
                token_max = m;
        }
        cluster.sync();
        amax = token_max;
#endif
    }

    float scale = quant_device::scale<Storage>(amax), inv = 1.f / scale;
    if (!threadIdx.x && !shard)
        scales[row] = scale;
#pragma unroll
    for (int j = 0; j < Items; ++j) {
        int c = (threadIdx.x + (j * Parts + shard) * Threads) * Vec;
        if (c < cols) {
            quant_device::BytePack<Vec> q;
#pragma unroll
            for (int p = 0; p < Vec / 2; ++p) {
                float2 f =
                    __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 *>(values[j].v + p * 2));
                q.v[p] = quant_device::Format<Storage>::pair(make_float2(f.x * inv, f.y * inv));
            }
            quant_device::store_pack(out + size_t(row) * cols + c, q, cols - c);
        }
    }
}
} // namespace tts::gated_kernels
