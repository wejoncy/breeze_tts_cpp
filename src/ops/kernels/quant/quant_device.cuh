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
template <> struct Packed16<float> {
    using Pair = float2;
    __device__ static float2 unpack(Pair v) { return v; }
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
// A power-of-two K group spans contiguous vector packs within one warp.
// Store scales in group-major order so producers need no layout conversion.
template <Type Storage, class T, int Count>
__device__ void encode_group(ValuePack<T, Count> value, unsigned char *output, float *scales,
                              int row, int column, int rows, int columns, int group_size) {
    using Math = Packed16<T>;
    using Pair = typename Math::Pair;
    float amax = 0;
#pragma unroll
    for (int pair = 0; pair < Count / 2; ++pair) {
        float2 elements = Math::unpack(*reinterpret_cast<Pair *>(value.v + 2 * pair));
        amax = fmaxf(amax, fmaxf(fabsf(elements.x), fabsf(elements.y)));
    }
    int lanes = group_size / Count, lane = threadIdx.x % 32;
    unsigned mask = (0xffffffffu >> (32 - lanes)) << (lane & ~(lanes - 1));
    amax = __uint_as_float(__reduce_max_sync(mask & __activemask(), __float_as_uint(amax)));
    float step = scale<Storage>(amax), inverse = 1.f / step;
    if (!(column & (group_size - 1)))
        scales[size_t(column >> (__ffs(group_size) - 1)) * rows + row] = step;
    BytePack<Count> encoded;
#pragma unroll
    for (int pair = 0; pair < Count / 2; ++pair) {
        float2 elements = Math::unpack(*reinterpret_cast<Pair *>(value.v + 2 * pair));
        encoded.v[pair] = Format<Storage>::pair(make_float2(elements.x * inverse,
                                                           elements.y * inverse));
    }
    store_pack(output + size_t(row) * columns + column, encoded, columns - column);
}
} // namespace tts::quant_device

#include <cooperative_groups.h>

namespace tts::quant_device {
// Matched Rademacher/Hadamard transforms preserve the dot product. The
// transform stays in registers inside the existing quantization producer.
template <class T, int Count>
__device__ ValuePack<float, Count> rotate_pack(ValuePack<T, Count> source, int column) {
    ValuePack<float, Count> values;
#pragma unroll
    for (int item = 0; item < Count; ++item) {
        unsigned hash = unsigned(column + item) * 0x45d9f3bu;
        hash = (hash ^ (hash >> 16)) * 0x45d9f3bu;
        values.v[item] = float(source.v[item]) * (((hash ^ (hash >> 16)) & 1) ? 1.f : -1.f);
    }
#pragma unroll
    for (int span = 1; span < Count; span *= 2) {
        auto previous = values;
#pragma unroll
        for (int item = 0; item < Count; ++item)
            values.v[item] = previous.v[item & ~span] +
                             ((item & span) ? -previous.v[item | span] : previous.v[item | span]);
    }
#pragma unroll
    for (int span = 1; span < 128 / Count; span *= 2) {
#pragma unroll
        for (int item = 0; item < Count; ++item) {
            float other = __shfl_xor_sync(0xffffffffu, values.v[item], span);
            values.v[item] = (threadIdx.x & span) ? other - values.v[item] : values.v[item] + other;
        }
    }
#pragma unroll
    for (int item = 0; item < Count; ++item)
        values.v[item] *= 0.08838834764831845f;
    return values;
}
template <int Threads, int Parts> __device__ float token_maximum(float value) {
    float amax = maximum<Threads>(value);
    if constexpr (Parts > 1) {
#if __CUDA_ARCH__ >= 900
        __shared__ float local_max, token_max;
        if (!threadIdx.x)
            local_max = amax;
        auto cluster = cooperative_groups::this_cluster();
        cluster.sync();
        if (threadIdx.x < 32) {
            float candidate =
                threadIdx.x < Parts ? *cluster.map_shared_rank(&local_max, threadIdx.x) : 0.f;
            candidate = __uint_as_float(__reduce_max_sync(0xffffffffu, __float_as_uint(candidate)));
            if (!threadIdx.x)
                token_max = candidate;
        }
        cluster.sync();
        amax = token_max;
#endif
    }
    return amax;
}
template <Type Storage, int Count>
__device__ void encode_rotated(ValuePack<float, Count> values, unsigned char *output, int column,
                               float inverse) {
    BytePack<Count> encoded;
#pragma unroll
    for (int pair = 0; pair < Count / 2; ++pair)
        encoded.v[pair] = Format<Storage>::pair(
            make_float2(values.v[2 * pair] * inverse, values.v[2 * pair + 1] * inverse));
    *reinterpret_cast<BytePack<Count> *>(output + column) = encoded;
}
template <Type Storage, int Threads, int Parts = 1, class T, int Count, int Items>
__device__ void rotated_quantize(ValuePack<T, Count> (&source)[Items], unsigned char *output,
                                 float *scales, int row, int columns, int shard = 0,
                                 int rows = 0, int group_size = 0) {
    ValuePack<float, Count> values[Items];
    float amax = 0;
#pragma unroll
    for (int item = 0; item < Items; ++item) {
        int column = (threadIdx.x + (item * Parts + shard) * Threads) * Count;
        values[item] = rotate_pack(source[item], column);
        if (!group_size) {
#pragma unroll
            for (int element = 0; element < Count; ++element)
                if (column < columns)
                    amax = fmaxf(amax, fabsf(values[item].v[element]));
        }
    }
    if (group_size) {
#pragma unroll
        for (int item = 0; item < Items; ++item) {
            int column = (threadIdx.x + (item * Parts + shard) * Threads) * Count;
            if (column < columns)
                encode_group<Storage>(values[item], output, scales, row, column,
                                      rows, columns, group_size);
        }
        return;
    }
    float step = scale<Storage>(token_maximum<Threads, Parts>(amax));
    if (!threadIdx.x && !shard)
        scales[row] = step;
#pragma unroll
    for (int item = 0; item < Items; ++item) {
        int column = (threadIdx.x + (item * Parts + shard) * Threads) * Count;
        if (column < columns)
            encode_rotated<Storage>(values[item], output + size_t(row) * columns, column,
                                    1.f / step);
    }
}
} // namespace tts::quant_device
