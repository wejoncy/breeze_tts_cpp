#pragma once
#include "kernels/quant/quant_format.cuh"
namespace tts::quant_device {
using bf16 = __nv_bfloat16;
template <int Count> struct alignas(Count * 2) Bf16Pack {
    bf16 v[Count];
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
