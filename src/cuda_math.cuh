#pragma once
#include <cuda_bf16.h>
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
__device__ inline float warp_sum(float x) {
    for (int i = 16; i; i /= 2)
        x += __shfl_down_sync(0xffffffff, x, i);
    return x;
}
} // namespace tts
