#include "ops/kernels/quant/quant_gated.h"
#include "ops/kernels/quant/quant_device.cuh"
#include <cuda_fp8.h>
#include <cfloat>
#include <type_traits>
#include <cooperative_groups.h>
namespace tts::gated_kernels {
using bf16 = __nv_bfloat16;
template <int Kind> __device__ __forceinline__ float activate(float x) {
    if constexpr (Kind == 0) {
        // Equal scaling keeps BF16 subnormals representable in FP32 and bounds
        // finite denominators within __fdividef's nonzero range (up to 2^126).
        return __fdividef(x * 0x1p-16f, (1.f + __expf(-x)) * 0x1p-16f);
    }
    return .5f * x * (1.f + tanhf(.7978845608028654f * (x + .044715f * x * x * x)));
}
// Preserve both storage rounding points; retain 16-bit pairs across token amax.
template <int Kind, int Threads, int Vec, int Parts = 1, Type Storage = Type::FP8, class T = bf16,
          bool Transformed = false>
__global__ void vector_gated_quantize(const T *input, unsigned char *out, float *scales, int cols,
                                      int rows = 0, int group_size = 0, int rotation_size = 0) {
    constexpr int Items = (8192 + Threads * Vec * Parts - 1) / (Threads * Vec * Parts);
    using Pack = quant_device::ValuePack<T, Vec>;
    using Math = quant_device::Packed16<T>;
    using Pair = typename Math::Pair;
    int row = blockIdx.x / Parts;
    int shard = blockIdx.x % Parts;
    Pack values[Items];
    float amax = 0;
#pragma unroll
    for (int j = 0; j < Items; ++j) {
        int c = (threadIdx.x + (j * Parts + shard) * Threads) * Vec;
        Pack value{};
        if (c < cols) {
            Pack g = quant_device::load_pack<Pack>(input + size_t(row) * cols * 2 + c, cols - c);
            Pack u =
                quant_device::load_pack<Pack>(input + size_t(row) * cols * 2 + cols + c, cols - c);
#pragma unroll
            for (int p = 0; p < Vec / 2; ++p) {
                float2 f = Math::unpack(*reinterpret_cast<Pair *>(g.v + p * 2));
                auto activated = Math::pack(activate<Kind>(f.x), activate<Kind>(f.y));
                auto v = __hmul2(activated, *reinterpret_cast<Pair *>(u.v + p * 2));
                *reinterpret_cast<Pair *>(value.v + p * 2) = v;
                if constexpr (!Transformed) {
                    f = Math::unpack(v);
                    amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
                }
            }
        }
        values[j] = value;
    }
    if constexpr (Transformed) {
        if (rotation_size) {
            quant_device::rotated_quantize<Storage, Threads, Parts>(values, out, scales, row, cols,
                                                                    shard, rows, group_size);
            return;
        }
#pragma unroll
        for (int item = 0; item < Items; ++item) {
            int column = (threadIdx.x + (item * Parts + shard) * Threads) * Vec;
            if (column < cols)
                quant_device::encode_group<Storage>(values[item], out, scales, row, column, rows,
                                                    cols, group_size);
        }
        return;
    }
    amax = quant_device::token_maximum<Threads, Parts>(amax);

    float scale = quant_device::scale<Storage>(amax), inv = 1.f / scale;
    if (!threadIdx.x && !shard)
        scales[row] = scale;
#pragma unroll
    for (int j = 0; j < Items; ++j) {
        int c = (threadIdx.x + (j * Parts + shard) * Threads) * Vec;
        if (c < cols) {
            quant_device::BytePack<Vec> q;
#pragma unroll
            for (int p = 0; p < Vec / 2; ++p) {
                float2 f = Math::unpack(*reinterpret_cast<Pair *>(values[j].v + p * 2));
                q.v[p] = quant_device::Format<Storage>::pair(make_float2(f.x * inv, f.y * inv));
            }
            quant_device::store_pack(out + size_t(row) * cols + c, q, cols - c);
        }
    }
}
} // namespace tts::gated_kernels
#include "ops/cuda/cuda_check.h"
namespace tts::gated_dispatch {
namespace {
// Keep host launch wrappers local to the CUDA module they were compiled with.
// Otherwise another executable/DSO can interpose a different-architecture image.
inline bool supports_clusters(int device) {
    int supported = 0;
    CUDA(cudaDeviceGetAttribute(&supported, cudaDevAttrClusterLaunch, device));
    if (!supported)
        return false;
    cudaFuncAttributes compiled{};
    CUDA(cudaFuncGetAttributes(&compiled,
                               gated_kernels::vector_gated_quantize<0, 256, 4, 2, Type::I8>));
    // PTX compiled below SM90 omits the cross-CTA reduction, even when JITed on Hopper.
    return compiled.ptxVersion >= 90;
}
template <int Kind, int Threads, int V, int Parts, Type Storage, class T>
void run(const T *x, unsigned char *q, float *s, int rows, int width, cudaStream_t stream,
         int group_size = 0, int rotation_size = 0) {
    auto execute = [&](auto transformation) {
        constexpr bool Transformed = decltype(transformation)::value;
        if (group_size) {
            gated_kernels::vector_gated_quantize<Kind, Threads, V, Parts, Storage, T, Transformed>
                <<<rows * Parts, Threads, 0, stream>>>(x, q, s, width, rows, group_size,
                                                       rotation_size);
            return;
        }
        if constexpr (Parts == 1)
            gated_kernels::vector_gated_quantize<Kind, Threads, V, 1, Storage, T, Transformed>
                <<<rows, Threads, 0, stream>>>(x, q, s, width, rows, 0, rotation_size);
        else {
            cudaLaunchConfig_t cfg{};
            cfg.gridDim = dim3(rows * Parts);
            cfg.blockDim = dim3(Threads);
            cfg.stream = stream;
            cudaLaunchAttribute attr{};
            attr.id = cudaLaunchAttributeClusterDimension;
            attr.val.clusterDim = {Parts, 1, 1};
            cfg.attrs = &attr;
            cfg.numAttrs = 1;
            CUDA(cudaLaunchKernelEx(&cfg,
                                    gated_kernels::vector_gated_quantize<Kind, Threads, V, Parts,
                                                                         Storage, T, Transformed>,
                                    x, q, s, width, rows, 0, rotation_size));
        }
    };
    if (group_size || rotation_size)
        execute(std::true_type{});
    else
        execute(std::false_type{});
}
} // namespace
} // namespace tts::gated_dispatch

namespace tts {
using bf16 = __nv_bfloat16;
namespace {
template <Type Storage, int Kind, class T>
void launch(TensorView input, RowScaledView output, int parts, cudaStream_t stream) {
#define RUN(P)                                                                                     \
    case P:                                                                                        \
        gated_dispatch::run<Kind, 256, P == 1 ? 8 : 4, P, Storage>(                                \
            input.ptr<T>(), output.values.ptr<unsigned char>(), output.scales.ptr<float>(),        \
            input.rows, output.values.cols, stream, output.group_size, output.rotation_size);      \
        break
    switch (parts) {
        RUN(1);
        RUN(2);
        RUN(4);
        RUN(8);
    }
#undef RUN
}
template <class T>
void launch_format(Type format, int kind, TensorView input, RowScaledView output, int parts,
                   cudaStream_t stream) {
    if (format == Type::I8) {
        if (kind == 0)
            launch<Type::I8, 0, T>(input, output, parts, stream);
        else
            launch<Type::I8, 1, T>(input, output, parts, stream);
    } else {
        if (kind == 0)
            launch<Type::FP8, 0, T>(input, output, parts, stream);
        else
            launch<Type::FP8, 1, T>(input, output, parts, stream);
    }
}
} // namespace
bool quant_gated_supports_clusters(int device) {
    return gated_dispatch::supports_clusters(device);
}
void launch_quantized_gated(TensorView input, RowScaledView output, int kind, cudaStream_t stream,
                            int multiprocessors, bool clusters) {
    output.validate();
    int width = output.values.cols;
    if ((input.type != Type::BF16 && input.type != Type::F16) || !input.data ||
        !output.values.data || !output.scales.data || !input.dense() ||
        input.rows != output.values.rows || input.cols != int64_t(2) * width || kind < 0 ||
        kind > 1 || width < 1 || width > 8192 || input.rows < 1)
        throw std::invalid_argument("quantized gated shape/type mismatch");
    int parts = 1;
    // Independent CTAs use 128-bit input loads. Clusters use 64-bit packs so
    // that short batches can spread the same token across more SMs.
    if (output.group_size || clusters)
        while (parts < 8 && parts * 2 <= multiprocessors / input.rows && 256 * 4 * parts < width)
            parts *= 2;
    if (input.type == Type::F16)
        launch_format<__half>(output.values.type, kind, input, output, parts, stream);
    else
        launch_format<bf16>(output.values.type, kind, input, output, parts, stream);
}
} // namespace tts
