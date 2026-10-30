#pragma once
#include "core/tensor.h"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cfloat>
#include <cmath>
namespace tts::quant_device {
template <Type Storage> struct Format;
template <> struct Format<Type::FP8> {
    using Acc = float;
    static constexpr float range = 448.f;
    __host__ __device__ static unsigned char encode(float x) {
        return __nv_fp8_e4m3(x).__x;
    }
    __device__ static unsigned short pair(float2 x) {
        return __nv_fp8x2_e4m3(x).__x;
    }
    __host__ __device__ static float decode(unsigned char x) {
        __nv_fp8_e4m3 v;
        v.__x = x;
        return float(v);
    }
    __device__ static float dot(unsigned int a, unsigned int b, float sum) {
        __nv_fp8x4_e4m3 av, bv;
        av.__x = a;
        bv.__x = b;
        float4 x = float4(av), y = float4(bv);
        sum = fmaf(x.x, y.x, sum);
        sum = fmaf(x.y, y.y, sum);
        sum = fmaf(x.z, y.z, sum);
        return fmaf(x.w, y.w, sum);
    }
};
template <> struct Format<Type::I8> {
    using Acc = int32_t;
    static constexpr float range = 127.f;
    __host__ __device__ static int rounded(float x) {
        x = fminf(127.f, fmaxf(-127.f, x));
#ifdef __CUDA_ARCH__
        return __float2int_rn(x);
#else
        return int(std::nearbyint(x));
#endif
    }
    __host__ __device__ static unsigned char encode(float x) {
        return static_cast<unsigned char>(rounded(x));
    }
    __device__ static unsigned short pair(float2 x) {
        unsigned packed;
        asm("cvt.pack.sat.s8.s32.b32 %0, %2, %1, 0;"
            : "=r"(packed)
            : "r"(rounded(x.x)), "r"(rounded(x.y)));
        return static_cast<unsigned short>(packed);
    }
    __host__ __device__ static float decode(unsigned char x) {
        return float(int8_t(x));
    }
    __device__ static int32_t dot(unsigned int a, unsigned int b, int32_t sum) {
        return __dp4a(int(a), int(b), sum);
    }
};
template <Type Storage> __device__ float scale(float amax) {
    return amax == 0 ? 1.f : fmaxf(amax / Format<Storage>::range, FLT_MIN);
}
} // namespace tts::quant_device
