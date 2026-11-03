#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <type_traits>
namespace tts {
using bf16 = __nv_bfloat16;
template <class T> __device__ float f(T x) {
    return float(x);
}
template <class T> __device__ T r(float x) {
    return T(x);
}
template <class T> __device__ float rounded(float x) {
    return f(r<T>(x));
}
// Preserve four consecutive FMA operands while loading one aligned vector.
template <class T> __device__ inline float4 load_four(const T *p) {
    if constexpr (std::is_same_v<T, float>) {
        return *reinterpret_cast<const float4 *>(p);
    } else {
        uint2 bits = *reinterpret_cast<const uint2 *>(p);
        float2 low, high;
        if constexpr (std::is_same_v<T, __half>) {
            low = __half22float2(*reinterpret_cast<const __half2 *>(&bits.x));
            high = __half22float2(*reinterpret_cast<const __half2 *>(&bits.y));
        } else {
            low = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&bits.x));
            high = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&bits.y));
        }
        return make_float4(low.x, low.y, high.x, high.y);
    }
}
template <class T> __device__ inline float warp_dot_four(const T *x, const T *w, int columns,
                                                        int lane, bool packed = true) {
    bool aligned = packed && ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(w)) &
                    (4 * sizeof(T) - 1)) == 0;
    float sum = 0;
    for (int c = lane * 4; c < columns; c += 128) {
        if (aligned && c + 3 < columns) {
            float4 a = load_four(x + c), b = load_four(w + c);
            sum = fmaf(b.x, a.x, sum);
            sum = fmaf(b.y, a.y, sum);
            sum = fmaf(b.z, a.z, sum);
            sum = fmaf(b.w, a.w, sum);
        } else {
#pragma unroll
            for (int j = 0; j < 4; ++j)
                if (c + j < columns) sum = fmaf(f(w[c + j]), f(x[c + j]), sum);
        }
    }
    return sum;
}
__device__ inline float warp_sum(float x) {
    for (int i = 16; i; i /= 2)
        x += __shfl_down_sync(0xffffffff, x, i);
    return x;
}
} // namespace tts
