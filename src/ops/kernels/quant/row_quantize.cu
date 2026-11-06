#include "ops/kernels/quant/row_quantize.h"
#include "ops/kernels/quant/quant_device.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <type_traits>

namespace tts {
using bf16 = __nv_bfloat16;
namespace {
template <int Threads = 256> __device__ float reduce_max(float value) {
    __shared__ float partial[Threads / 32];
    for (int d = 16; d; d /= 2)
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, d));
    if (!(threadIdx.x % 32))
        partial[threadIdx.x / 32] = value;
    __syncthreads();
    int lane = threadIdx.x % 32;
    float result = lane < Threads / 32 ? partial[lane] : 0.f;
    for (int d = 16; d; d /= 2)
        result = fmaxf(result, __shfl_down_sync(0xffffffff, result, d));
    return __shfl_sync(0xffffffff, result, 0);
}
template <Type Storage, class T>
__global__ void dynamic_token_quantize(const T *x, unsigned char *q, float *scales, int cols,
                                       int stride) {
    int row = blockIdx.x;
    float amax = 0;
    for (int c = threadIdx.x; c < cols; c += 256)
        amax = fmaxf(amax, fabsf(float(x[size_t(row) * stride + c])));
    float scale = quant_device::scale<Storage>(reduce_max(amax)), inv = 1.f / scale;
    if (!threadIdx.x)
        scales[row] = scale;
    for (int c = threadIdx.x; c < cols; c += 256)
        q[size_t(row) * cols + c] =
            quant_device::Format<Storage>::encode(float(x[size_t(row) * stride + c]) * inv);
}
// Cache sixteen 16-bit values per thread; wider or unaligned rows keep the
// scalar path. Selection depends only on layout and register-tile capacity.
constexpr int token_quantize_threads = 256;
constexpr int token_quantize_values = 16;
template <Type Format, class T>
__global__ void cached_token_quantize(const T *x, unsigned char *q, float *scales,
                                      int cols, int stride) {
    using Pack = quant_device::ValuePack<T, 4>;
    using Pair = typename quant_device::Packed16<T>::Pair;
    int row = blockIdx.x;
    x += size_t(row) * stride;
    q += size_t(row) * cols;
    constexpr int packets = token_quantize_values / 4;
    Pack values[packets];
    float amax = 0;
#pragma unroll
    for (int i = 0; i < packets; ++i) {
        int c = threadIdx.x * 4 + i * token_quantize_threads * 4;
        Pack value{};
        if (c < cols)
            value = *reinterpret_cast<const Pack *>(x + c);
        values[i] = value;
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            float2 v = quant_device::Packed16<T>::unpack(
                *reinterpret_cast<const Pair *>(value.v + p * 2));
            amax = fmaxf(amax, fmaxf(fabsf(v.x), fabsf(v.y)));
        }
    }
    float scale = quant_device::scale<Format>(quant_device::maximum<token_quantize_threads>(amax));
    float inverse = 1.f / scale;
    if (!threadIdx.x)
        scales[row] = scale;
#pragma unroll
    for (int i = 0; i < packets; ++i) {
        int c = threadIdx.x * 4 + i * token_quantize_threads * 4;
        if (c < cols) {
            Pack value = values[i];
            quant_device::BytePack<4> encoded;
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                float2 v = quant_device::Packed16<T>::unpack(
                    *reinterpret_cast<const Pair *>(value.v + p * 2));
                encoded.v[p] = quant_device::Format<Format>::pair(
                    make_float2(v.x * inverse, v.y * inverse));
            }
            *reinterpret_cast<quant_device::BytePack<4> *>(q + c) = encoded;
        }
    }
}
// Two register passes cover arbitrary K without an intermediate tensor.
// This also performs the weight transform once during loading.
template <Type Storage, class T>
__global__ void rotated_token_quantize(const T *input, unsigned char *output, float *scales,
                                        int columns, int stride) {
    using Pack = quant_device::ValuePack<T, 8>;
    int row = blockIdx.x;
    float amax = 0;
    for (int column = threadIdx.x * 8; column < (columns + 2047) / 2048 * 2048; column += 2048) {
        Pack source{};
        if (column < columns) source = quant_device::load_pack<Pack>(
            input + size_t(row) * stride + column, columns - column);
        auto values = quant_device::rotate_pack(source, column);
#pragma unroll
        for (int item = 0; item < 8; ++item) amax = fmaxf(amax, fabsf(values.v[item]));
    }
    float step = quant_device::scale<Storage>(quant_device::maximum<256>(amax));
    if (!threadIdx.x) scales[row] = step;
    for (int column = threadIdx.x * 8; column < (columns + 2047) / 2048 * 2048; column += 2048) {
        Pack source{};
        if (column < columns) source = quant_device::load_pack<Pack>(
            input + size_t(row) * stride + column, columns - column);
        auto values = quant_device::rotate_pack(source, column);
        if (column < columns)
            quant_device::encode_rotated<Storage>(values, output + size_t(row) * columns,
                                                   column, 1.f / step);
    }
}
template <class T>
void quantize_rows(Type format, const T *x, unsigned char *q, float *scales, int rows, int cols,
                   int stride, cudaStream_t stream, int rotation_size) {
    if (rotation_size) {
        if (format == Type::I8)
            rotated_token_quantize<Type::I8><<<rows, 256, 0, stream>>>(x, q, scales, cols, stride);
        else
            rotated_token_quantize<Type::FP8><<<rows, 256, 0, stream>>>(x, q, scales, cols, stride);
        return;
    }
    using Pack = quant_device::ValuePack<T, 4>;
    bool cached = cols <= token_quantize_threads * token_quantize_values &&
                  cols % 4 == 0 && stride % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(x) % alignof(Pack) == 0 &&
                  reinterpret_cast<uintptr_t>(q) % alignof(quant_device::BytePack<4>) == 0;
    if (format == Type::I8) {
        if (cached)
            cached_token_quantize<Type::I8><<<rows, token_quantize_threads, 0, stream>>>(
                x, q, scales, cols, stride);
        else
            dynamic_token_quantize<Type::I8, T><<<rows, 256, 0, stream>>>(x, q, scales, cols, stride);
    } else {
        if (cached)
            cached_token_quantize<Type::FP8><<<rows, token_quantize_threads, 0, stream>>>(
                x, q, scales, cols, stride);
        else
            dynamic_token_quantize<Type::FP8, T><<<rows, 256, 0, stream>>>(x, q, scales, cols, stride);
    }
}
template <Type Format, class T>
__global__ void group_token_quantize(const T *input, unsigned char *output, float *scales,
                                      int rows, int columns, int stride, int group_size,
                                      int rotation_size) {
    constexpr int width = 256 * 4;
    int tiles = (columns + width - 1) / width;
    int row = blockIdx.x / tiles;
    int column = (blockIdx.x % tiles * 256 + threadIdx.x) * 4;
    if (column < columns) {
        using Pack = quant_device::ValuePack<T, 4>;
        auto value = quant_device::load_pack<Pack>(input + size_t(row) * stride + column,
                                                   columns - column);
        if (rotation_size) {
            auto rotated = quant_device::rotate_pack(value, column);
            quant_device::encode_group<Format>(rotated, output, scales, row, column,
                                               rows, columns, group_size);
        } else
            quant_device::encode_group<Format>(value, output, scales, row, column,
                                               rows, columns, group_size);
    }
}
} // namespace
void launch_row_quantize(TensorView source, RowScaledView destination, cudaStream_t stream) {
    if (destination.group_size) {
        auto launch = [&](auto value, auto format) {
            using T = decltype(value);
            constexpr Type storage = decltype(format)::value;
            uint64_t blocks = uint64_t(source.rows) * ((uint64_t(source.cols) + 1023) / 1024);
            if (blocks > INT32_MAX) throw std::invalid_argument("group quantization grid exceeds capacity");
            group_token_quantize<storage><<<unsigned(blocks), 256, 0, stream>>>(
                source.ptr<T>(), destination.values.ptr<unsigned char>(), destination.scales.ptr<float>(),
                source.rows, source.cols, source.ld(), destination.group_size,
                destination.rotation_size);
        };
        auto format = [&](auto value) {
            if (destination.values.type == Type::I8) launch(value, std::integral_constant<Type, Type::I8>{});
            else launch(value, std::integral_constant<Type, Type::FP8>{});
        };
        if (source.type == Type::F16) format(__half{});
        else format(bf16{});
    } else if (source.type == Type::F16)
        quantize_rows(destination.values.type, source.ptr<__half>(), destination.values.ptr<unsigned char>(),
                      destination.scales.ptr<float>(), source.rows, source.cols, source.ld(),
                      stream, destination.rotation_size);
    else
        quantize_rows(destination.values.type, source.ptr<bf16>(), destination.values.ptr<unsigned char>(),
                      destination.scales.ptr<float>(), source.rows, source.cols, source.ld(),
                      stream, destination.rotation_size);
}
} // namespace tts
