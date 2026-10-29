#pragma once
#include "quant_format.cuh"
namespace tts::quant_device {
using bf16 = __nv_bfloat16;
template <int Count> struct alignas(Count * 2) Bf16Pack {
    bf16 v[Count];
};
template <int Count> struct alignas(Count) BytePack {
    unsigned short v[Count / 2];
};
template <int Threads> __device__ float maximum(float v) {
    __shared__ unsigned partial[Threads / 32];
    unsigned bits = __reduce_max_sync(0xffffffffu, __float_as_uint(v));
    if (!(threadIdx.x % 32))
        partial[threadIdx.x / 32] = bits;
    __syncthreads();
    unsigned b = threadIdx.x % 32 < Threads / 32 ? partial[threadIdx.x % 32] : 0u;
    return __uint_as_float(__reduce_max_sync(0xffffffffu, b));
}
} // namespace tts::quant_device
