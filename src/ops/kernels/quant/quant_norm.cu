#include "ops/kernels/quant/quant_norm.h"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cfloat>
#include <type_traits>
#include "ops/kernels/quant/quant_device.cuh"

namespace tts::norm_kernels {
using bf16 = __nv_bfloat16;
using quant_device::BytePack;
using quant_device::maximum;
// Vector loads and storage pair arithmetic preserve the residual and normalized
// rounding points. Weights are loaded before the sum reduction to hide latency.
template <int Threads, int Vec, bool Add, Type Storage = Type::FP8, class T = bf16,
          bool Transformed = false>
__global__ void residual_rms_quantize(T *x, const T *update, const T *weight, const T *update_scale,
                                      unsigned char *out, float *scales, int cols, float eps,
                                      bool gemma, int rows, int group_size, int rotation_size = 0) {
    constexpr int Items = (2048 + Threads * Vec - 1) / (Threads * Vec);
    using Pack = quant_device::ValuePack<T, Vec>;
    using Math = quant_device::Packed16<T>;
    using Pair = typename Math::Pair;
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
                    auto a = *reinterpret_cast<Pair *>(v.v + p * 2);
                    auto b = *reinterpret_cast<Pair *>(u.v + p * 2);
                    if (update_scale)
                        b = __hmul2(b, *reinterpret_cast<Pair *>(s.v + p * 2));
                    *reinterpret_cast<Pair *>(v.v + p * 2) = __hadd2(a, b);
                }
                quant_device::store_pack(x + size_t(row) * cols + c, v, cols - c);
            }
        }
        vals[j] = v;
        weights[j] = w;
#pragma unroll
        for (int p = 0; p < Vec / 2; ++p) {
            float2 f = Math::unpack(*reinterpret_cast<Pair *>(v.v + p * 2));
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
                auto *v = reinterpret_cast<Pair *>(vals[j].v + p * 2);
                auto w = *reinterpret_cast<Pair *>(weights[j].v + p * 2);
                float2 f = Math::unpack(*v);
                if (gemma) {
                    float2 wf = Math::unpack(w);
                    *v = Math::pack((f.x * inv_norm) * (wf.x + 1.f),
                                    (f.y * inv_norm) * (wf.y + 1.f));
                } else {
                    *v = __hmul2(Math::pack(f.x * inv_norm, f.y * inv_norm), w);
                }
                if constexpr (!Transformed) {
                    f = Math::unpack(*v);
                    amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
                }
            }
        }
    }
    if constexpr (Transformed) {
        if (rotation_size) {
            quant_device::rotated_quantize<Storage, Threads>(vals, out, scales, row, cols, 0, rows,
                                                             group_size);
            return;
        }
#pragma unroll
        for (int item = 0; item < Items; ++item) {
            int column = (threadIdx.x + item * Threads) * Vec;
            if (column < cols)
                quant_device::encode_group<Storage>(vals[item], out, scales, row, column, rows,
                                                    cols, group_size);
        }
        return;
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
                float2 f = Math::unpack(*reinterpret_cast<Pair *>(vals[j].v + p * 2));
                q.v[p] = quant_device::Format<Storage>::pair(make_float2(f.x * inv, f.y * inv));
            }
            quant_device::store_pack(out + size_t(row) * cols + c, q, cols - c);
        }
    }
}
} // namespace tts::norm_kernels

namespace tts {
namespace {
using bf16 = __nv_bfloat16;
template <Type Storage, class T>
void launch(T *residual, const T *update, const T *weight, const T *update_scale,
            unsigned char *values, float *scales, int rows, int width, float epsilon, bool gemma,
            int group_size, int rotation_size, cudaStream_t stream) {
    // Keep optional transforms out of the row-scaled kernel's register budget.
    auto execute = [&](auto transformation) {
        constexpr bool Transformed = decltype(transformation)::value;
        if (update)
            norm_kernels::residual_rms_quantize<256, 8, true, Storage, T, Transformed>
                <<<rows, 256, 0, stream>>>(residual, update, weight, update_scale, values, scales,
                                           width, epsilon, gemma, rows, group_size, rotation_size);
        else
            norm_kernels::residual_rms_quantize<256, 8, false, Storage, T, Transformed>
                <<<rows, 256, 0, stream>>>(residual, update, weight, update_scale, values, scales,
                                           width, epsilon, gemma, rows, group_size, rotation_size);
    };
    if (group_size || rotation_size)
        execute(std::true_type{});
    else
        execute(std::false_type{});
}
template <class T>
void launch_format(Type format, TensorView residual, TensorView update, TensorView weight,
                   TensorView update_scale, RowScaledView output, float epsilon, bool gemma,
                   cudaStream_t stream) {
    if (format == Type::I8)
        launch<Type::I8>(residual.ptr<T>(), update.ptr<T>(), weight.ptr<T>(), update_scale.ptr<T>(),
                         output.values.ptr<unsigned char>(), output.scales.ptr<float>(),
                         residual.rows, residual.cols, epsilon, gemma, output.group_size,
                         output.rotation_size, stream);
    else
        launch<Type::FP8>(residual.ptr<T>(), update.ptr<T>(), weight.ptr<T>(),
                          update_scale.ptr<T>(), output.values.ptr<unsigned char>(),
                          output.scales.ptr<float>(), residual.rows, residual.cols, epsilon, gemma,
                          output.group_size, output.rotation_size, stream);
}
} // namespace
void launch_quantized_norm(TensorView residual, TensorView update, TensorView weight,
                           RowScaledView output, float epsilon, bool gemma, TensorView update_scale,
                           cudaStream_t stream) {
    output.validate();
    if ((residual.type != Type::BF16 && residual.type != Type::F16) || !residual.data ||
        !weight.data || !output.values.data || !output.scales.data || !residual.dense() ||
        residual.cols < 1 || residual.cols > 2048 || residual.rows < 1 ||
        residual.cols != output.values.cols || residual.rows != output.values.rows ||
        weight.type != residual.type || !weight.dense() ||
        weight.numel() != size_t(residual.cols) ||
        (update.data && (update.type != residual.type || update.rows != residual.rows ||
                         update.cols != residual.cols || !update.dense())) ||
        (update_scale.data && (update_scale.type != residual.type || !update_scale.dense() ||
                               update_scale.numel() != size_t(residual.cols))))
        throw std::invalid_argument("quantized RMS shape/type mismatch");
    if (residual.type == Type::F16)
        launch_format<__half>(output.values.type, residual, update, weight, update_scale, output,
                              epsilon, gemma, stream);
    else
        launch_format<bf16>(output.values.type, residual, update, weight, update_scale, output,
                            epsilon, gemma, stream);
}
} // namespace tts
