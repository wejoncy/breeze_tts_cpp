#pragma once
#include "ops/kernels/quant/quant_format.cuh"
#include <cuda_fp16.h>
namespace tts::quant_device {
using bf16 = __nv_bfloat16;
template <class T, int Count> struct alignas(Count * sizeof(T)) ValuePack {
    T v[Count];
};
// Match the original storage rounding points while sharing packed arithmetic.
template <class T> struct Packed16;
template <> struct Packed16<bf16> {
    using Pair = __nv_bfloat162;
    __device__ static float2 unpack(Pair v) { return __bfloat1622float2(v); }
    __device__ static Pair pack(float x, float y) { return __floats2bfloat162_rn(x, y); }
};
template <> struct Packed16<__half> {
    using Pair = __half2;
    __device__ static float2 unpack(Pair v) { return __half22float2(v); }
    __device__ static Pair pack(float x, float y) { return __floats2half2_rn(x, y); }
};
template <int Count> struct alignas(Count) BytePack {
    unsigned short v[Count / 2];
};
template <class Pack, class T> __device__ Pack load_pack(const T *p, int remaining) {
    constexpr int count = sizeof(Pack) / sizeof(T);
    if (remaining >= count && reinterpret_cast<uintptr_t>(p) % alignof(Pack) == 0)
        return *reinterpret_cast<const Pack *>(p);
    Pack value{};
    auto *elements = reinterpret_cast<T *>(&value);
#pragma unroll
    for (int i = 0; i < count; ++i)
        if (i < remaining)
            elements[i] = p[i];
    return value;
}
template <class Pack, class T> __device__ void store_pack(T *p, Pack value, int remaining) {
    constexpr int count = sizeof(Pack) / sizeof(T);
    if (remaining >= count && reinterpret_cast<uintptr_t>(p) % alignof(Pack) == 0) {
        *reinterpret_cast<Pack *>(p) = value;
        return;
    }
    const auto *elements = reinterpret_cast<const T *>(&value);
#pragma unroll
    for (int i = 0; i < count; ++i)
        if (i < remaining)
            p[i] = elements[i];
}
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
