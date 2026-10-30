#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cfloat>
#include "kernels/quant/quant_device.cuh"

namespace tts::norm_kernels {
using bf16 = __nv_bfloat16;
using quant_device::Bf16Pack;
using quant_device::BytePack;
using quant_device::maximum;
// Vector loads and BF16 pair arithmetic preserve the residual and normalized
// rounding points. Weights are loaded before the sum reduction to hide latency.
template <int Threads, int Vec, bool Add, Type Storage = Type::FP8>
__global__ void residual_rms_quantize(bf16 *x, const bf16 *update, const bf16 *weight,
                                      const bf16 *update_scale, unsigned char *out, float *scales,
                                      int cols, float eps, bool gemma) {
    constexpr int Items = (2048 + Threads * Vec - 1) / (Threads * Vec);
    using Pack = Bf16Pack<Vec>;
    __shared__ float partial[Threads / 32];
    int row = blockIdx.x;
    Pack vals[Items], weights[Items];
    float sum = 0;
#pragma unroll
    for (int j = 0; j < Items; ++j) {
        int c = (threadIdx.x + j * Threads) * Vec;
        Pack v{}, w{};
        if (c < cols) {
            v = quant_device::load_pack<Pack>(x + size_t(row) * cols + c, cols - c);
            w = quant_device::load_pack<Pack>(weight + c, cols - c);
            if constexpr (Add) {
                Pack u = quant_device::load_pack<Pack>(update + size_t(row) * cols + c, cols - c),
                     s{};
                if (update_scale)
                    s = quant_device::load_pack<Pack>(update_scale + c, cols - c);
#pragma unroll
                for (int p = 0; p < Vec / 2; ++p) {
                    auto a = *reinterpret_cast<__nv_bfloat162 *>(v.v + p * 2);
                    auto b = *reinterpret_cast<__nv_bfloat162 *>(u.v + p * 2);
                    if (update_scale)
                        b = __hmul2(b, *reinterpret_cast<__nv_bfloat162 *>(s.v + p * 2));
                    *reinterpret_cast<__nv_bfloat162 *>(v.v + p * 2) = __hadd2(a, b);
                }
                quant_device::store_pack(x + size_t(row) * cols + c, v, cols - c);
            }
        }
        vals[j] = v;
        weights[j] = w;
#pragma unroll
        for (int p = 0; p < Vec / 2; ++p) {
            float2 f = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 *>(v.v + p * 2));
            sum += f.x * f.x + f.y * f.y;
        }
    }
    for (int d = 16; d; d /= 2)
        sum += __shfl_down_sync(0xffffffffu, sum, d);
    if (!(threadIdx.x % 32))
        partial[threadIdx.x / 32] = sum;
    __syncthreads();
    float total = 0;
#pragma unroll
    for (int p = 0; p < Threads / 32; ++p)
        total += partial[p];
    float inv_norm = rsqrtf(total / cols + eps), amax = 0;
#pragma unroll
    for (int j = 0; j < Items; ++j) {
        int c = (threadIdx.x + j * Threads) * Vec;
        if (c < cols) {
#pragma unroll
            for (int p = 0; p < Vec / 2; ++p) {
                auto *v = reinterpret_cast<__nv_bfloat162 *>(vals[j].v + p * 2);
                auto w = *reinterpret_cast<__nv_bfloat162 *>(weights[j].v + p * 2);
                float2 f = __bfloat1622float2(*v);
                if (gemma) {
                    float2 wf = __bfloat1622float2(w);
                    *v = __floats2bfloat162_rn((f.x * inv_norm) * (wf.x + 1.f),
                                               (f.y * inv_norm) * (wf.y + 1.f));
                } else {
                    *v = __hmul2(__floats2bfloat162_rn(f.x * inv_norm, f.y * inv_norm), w);
                }
                f = __bfloat1622float2(*v);
                amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
            }
        }
    }
    amax = maximum<Threads>(amax);
    float scale = quant_device::scale<Storage>(amax), inv = 1.f / scale;
    if (!threadIdx.x)
        scales[row] = scale;
#pragma unroll
    for (int j = 0; j < Items; ++j) {
        int c = (threadIdx.x + j * Threads) * Vec;
        if (c < cols) {
            BytePack<Vec> q;
#pragma unroll
            for (int p = 0; p < Vec / 2; ++p) {
                float2 f =
                    __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 *>(vals[j].v + p * 2));
                q.v[p] = quant_device::Format<Storage>::pair(make_float2(f.x * inv, f.y * inv));
            }
            quant_device::store_pack(out + size_t(row) * cols + c, q, cols - c);
        }
    }
}
} // namespace tts::norm_kernels
