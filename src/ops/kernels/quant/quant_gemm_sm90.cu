#include "ops/kernels/quant/quant_gemm.h"
#include "ops/kernels/quant/quant_reduce.cuh"
#include "ops/kernels/common/activation.cuh"
#include <stdexcept>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp>
#include <cuda_bf16.h>
#include <type_traits>
namespace tts {
namespace native_partition {
using namespace cute;
template <class E, int BM = 64, int BN = 16> struct Native {
    using Acc = std::conditional_t<std::is_same_v<E, int8_t>, int32_t, float>;
    using Tile = Shape<Int<BM>, Int<BN>, _128>;
    using Cluster = Shape<_1, _1, _1>;
    using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, Tile, Cluster,
        cutlass::epilogue::collective::EpilogueTileAuto, Acc, Acc, void,
        cutlass::layout::ColumnMajor, 4, Acc, cutlass::layout::ColumnMajor, 4,
        cutlass::epilogue::TmaWarpSpecialized>::CollectiveOp;
    using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, E, cutlass::layout::RowMajor, 16, E,
        cutlass::layout::ColumnMajor, 16, Acc, Tile, Cluster,
        cutlass::gemm::collective::StageCount<4>,
        cutlass::gemm::KernelTmaWarpSpecialized>::CollectiveOp;
    using Kernel =
        cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
    static void run(void *x, const float *sx, void *w, const float *sw, void *y, int m, int n,
                    int k, void *workspace, size_t bytes, cudaStream_t stream, int splits,
                    Type output_type) {
        if (k % splits || (k / splits) % 16)
            throw std::runtime_error("unaligned native K partition");
        if (size_t(m) * n * splits * 4 > bytes)
            throw std::runtime_error("native partial workspace overflow");
        typename Mainloop::Arguments mainloop{};
        if constexpr (std::is_same_v<Acc, float>)
            mainloop.mma_promotion_interval = 4;
        mainloop.ptr_A = static_cast<E *>(w);
        mainloop.dA = {k, Int<1>{}, k / splits};
        mainloop.ptr_B = static_cast<E *>(x);
        mainloop.dB = {k, Int<1>{}, k / splits};
        typename Epilogue::Arguments epilogue{};
        epilogue.thread = {1, 0};
        epilogue.ptr_D = static_cast<Acc *>(workspace);
        epilogue.dD = {Int<1>{}, n, n * m};
        epilogue.dC = {Int<1>{}, n, n * m};
        typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kBatched,
                                      {n, m, k / splits, splits},
                                      mainloop,
                                      epilogue};
        if (Gemm::get_workspace_size(args))
            throw std::runtime_error("unexpected native workspace");
        Gemm gemm;
        auto check = [](cutlass::Status s) {
            if (s != cutlass::Status::kSuccess)
                throw std::runtime_error(cutlassGetStatusString(s));
        };
        check(gemm.can_implement(args));
        check(gemm.initialize(args, nullptr, stream));
        check(gemm.run(stream));
        launch_quant_split_reduce(static_cast<Acc *>(workspace), sx, sw, y, m, n, splits, stream,
                                  output_type);
    }
};
} // namespace native_partition

namespace native_direct {
using namespace cute;
namespace fusion = cutlass::epilogue::fusion;
struct ScaleOnly {
    template <class Scaled, class Tile, class D> struct For {
        using EVT = Scaled;
        static typename EVT::Arguments arguments(const float *sw, const float *sx,
                                                 const void *, ops::Activation) {
            return {{sw}, {{sx}, {}, {}}, {}};
        }
    };
};
struct ActivationArguments { ops::Activation kind; };
template <class Fragment> struct RuntimeActivation {
    using Arguments = ActivationArguments;
    CUTLASS_DEVICE Fragment operator()(Fragment const &input, Arguments const &args) const {
        Fragment output;
        // Dispatch once per fragment; each branch reuses the scalar formula
        // with a constant kind, without multiplying kernel instantiations.
        switch (args.kind) {
#define APPLY(A) case ops::Activation::A: \
            CUTLASS_PRAGMA_UNROLL \
            for (int i = 0; i < Fragment::kElements; ++i) \
                output[i] = ops::activate(input[i], ops::Activation::A, 0.f); \
            break
        APPLY(Relu); APPLY(GeluTanh);
#undef APPLY
        default: output = input; break;
        }
        return output;
    }
};
struct FinishPolicy {
    template <class Scaled, class Tile, class D> struct For {
        using Bias = fusion::Sm90ColBroadcast<0, Tile, D, float, Stride<_1, _0, _0>>;
        using Add = fusion::Sm90Compute<cutlass::plus, D, float,
                                        cutlass::FloatRoundStyle::round_to_nearest>;
        using WithBias = fusion::Sm90EVT<Add, Bias, Scaled>;
        using Act = fusion::Sm90Compute<RuntimeActivation, D, float,
                                        cutlass::FloatRoundStyle::round_to_nearest>;
        using EVT = fusion::Sm90EVT<Act, WithBias>;
        static typename EVT::Arguments arguments(const float *sw, const float *sx,
                                                 const void *bias, ops::Activation kind) {
            typename Scaled::Arguments scales{{sw}, {{sx}, {}, {}}, {}};
            typename WithBias::Arguments biased{{static_cast<D const *>(bias)}, scales, {}};
            return {biased, {kind}};
        }
    };
};
template <class E, class D, int BM = 64, int BN = 16, class Finish = ScaleOnly> struct Native {
    using Acc = std::conditional_t<std::is_same_v<E, int8_t>, int32_t, float>;
    using Tile = Shape<Int<BM>, Int<BN>, _128>;
    using Cluster = Shape<_1, _1, _1>;
    using ScaleW = fusion::Sm90ColBroadcast<0, Tile, float, float, Stride<_1, _0, _0>>;
    using ScaleX = fusion::Sm90RowBroadcast<0, Tile, float, float, Stride<_0, _1, _0>>;
    using Mul = fusion::Sm90Compute<cutlass::multiplies, float, float,
                                    cutlass::FloatRoundStyle::round_to_nearest>;
    using MulOut = fusion::Sm90Compute<cutlass::multiplies, D, float,
                                       cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT0 = fusion::Sm90EVT<Mul, ScaleX, fusion::Sm90AccFetch>;
    using Scaled = fusion::Sm90EVT<MulOut, ScaleW, EVT0>;
    using Policy = typename Finish::template For<Scaled, Tile, D>;
    using EVT = typename Policy::EVT;
    using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, Tile, Cluster,
        cutlass::epilogue::collective::EpilogueTileAuto, Acc, float, void,
        cutlass::layout::ColumnMajor, 8, D, cutlass::layout::ColumnMajor, 8,
        cutlass::epilogue::TmaWarpSpecialized, EVT>::CollectiveOp;
    using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, E, cutlass::layout::RowMajor, 16, E,
        cutlass::layout::ColumnMajor, 16, Acc, Tile, Cluster,
        cutlass::gemm::collective::StageCount<4>,
        cutlass::gemm::KernelTmaWarpSpecialized>::CollectiveOp;
    using Kernel =
        cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
    static KernelResources resources() {
        cudaFuncAttributes attributes{};
        auto status = cudaFuncGetAttributes(&attributes, cutlass::device_kernel<Kernel>);
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        int resident = Gemm::maximum_active_blocks();
        if (resident < 1) throw std::runtime_error("native GEMM has no resident CTA");
        return {attributes.numRegs, resident};
    }
    static void run(void *x, const float *sx, void *w, const float *sw, void *y, int m, int n,
                    int k, void *workspace, size_t bytes, cudaStream_t stream,
                    const void *bias = nullptr, ops::Activation activation = ops::Activation::Identity) {
        typename Mainloop::Arguments mainloop{};
        if constexpr (std::is_same_v<Acc, float>)
            mainloop.mma_promotion_interval = 4;
        mainloop.ptr_A = static_cast<E *>(w);
        mainloop.dA = {k, Int<1>{}, n * k};
        mainloop.ptr_B = static_cast<E *>(x);
        mainloop.dB = {k, Int<1>{}, m * k};
        typename Epilogue::Arguments epilogue{};
        epilogue.thread = Policy::arguments(sw, sx, bias, activation);
        epilogue.ptr_D = static_cast<D *>(y);
        epilogue.dD = {Int<1>{}, n, n * m};
        epilogue.dC = {Int<1>{}, n, n * m};
        typename Gemm::Arguments args{
            cutlass::gemm::GemmUniversalMode::kGemm, {n, m, k, 1}, mainloop, epilogue};
        if (Gemm::get_workspace_size(args) > bytes)
            throw std::runtime_error("CUTLASS workspace overflow");
        Gemm gemm;
        auto check = [](cutlass::Status s) {
            if (s != cutlass::Status::kSuccess)
                throw std::runtime_error(cutlassGetStatusString(s));
        };
        check(gemm.can_implement(args));
        check(gemm.initialize(args, workspace, stream));
        check(gemm.run(stream));
    }
};
// One geometry dispatch for execution and initialization resource inspection.
template <class E, class D, class Policy, class Visit>
void visit(int channels, int tokens, Visit &&invoke) {
#define APPLY(C, T) invoke(Native<E, D, C, T, Policy>{})
    if (channels == 64 && tokens == 16) { APPLY(64, 16); }
    else if (channels == 128 && tokens == 16) { APPLY(128, 16); }
    else if (channels == 64 && tokens == 32) { APPLY(64, 32); }
    else if (channels == 128 && tokens == 32) { APPLY(128, 32); }
    else if (channels == 64 && tokens == 64) { APPLY(64, 64); }
    else if (channels == 128 && tokens == 64) { APPLY(128, 64); }
    else if (channels == 64 && tokens == 128) { APPLY(64, 128); }
    else if (channels == 128 && tokens == 128) {
        if constexpr (std::is_same_v<E, int8_t>) { APPLY(128, 128); }
        else throw std::invalid_argument("FP8 tile exceeds consumer register capacity");
    } else throw std::invalid_argument("invalid quantized tile");
#undef APPLY
}
} // namespace native_direct

} // namespace tts

namespace tts {
template <class E, class D, class Finish = native_direct::ScaleOnly>
void hopper_gemm(void *x, const float *sx, void *w, const float *sw, void *y, int rows, int n,
                 int k, void *workspace, size_t bytes, cudaStream_t stream, int tile_channels,
                 int tile_tokens, int splits, Type output_type,
                 const void *bias = nullptr, ops::Activation activation = ops::Activation::Identity) {
#define RUN_SPLIT(N)                                                                               \
    native_partition::Native<E, 64, N>::run(x, sx, w, sw, y, rows, n, k, workspace, bytes, stream, \
                                            splits, output_type)
    if constexpr (!std::is_same_v<Finish, native_direct::ScaleOnly>)
        if (splits != 1) throw std::invalid_argument("native Finish does not support split K");
    if (splits == 1) {
        native_direct::visit<E, D, Finish>(tile_channels, tile_tokens, [&](auto kernel) {
            decltype(kernel)::run(x, sx, w, sw, y, rows, n, k, workspace, bytes, stream, bias, activation);
        });
    } else {
        if (tile_channels != 64)
            throw std::invalid_argument("invalid quantized split tile");
        if (tile_tokens == 16) {
            RUN_SPLIT(16);
        } else if (tile_tokens == 32) {
            RUN_SPLIT(32);
        } else if (tile_tokens == 64) {
            RUN_SPLIT(64);
        } else if (tile_tokens == 128) {
            RUN_SPLIT(128);
        } else
            throw std::invalid_argument("invalid quantized token tile");
    }
#undef RUN_SPLIT
}
template <class E, class Finish = native_direct::ScaleOnly>
void typed_hopper_gemm(void *x, const float *sx, void *w, const float *sw, void *y, int rows, int n,
                       int k, void *workspace, size_t bytes, cudaStream_t stream, int channels,
                       int tokens, int splits, Type output_type,
                       const void *bias = nullptr, ops::Activation activation = ops::Activation::Identity) {
    if (output_type == Type::F16)
        hopper_gemm<E, cutlass::half_t, Finish>(x, sx, w, sw, y, rows, n, k, workspace, bytes, stream,
                                        channels, tokens, splits, output_type, bias, activation);
    else if (output_type == Type::BF16)
        hopper_gemm<E, cutlass::bfloat16_t, Finish>(x, sx, w, sw, y, rows, n, k, workspace, bytes, stream,
                                            channels, tokens, splits, output_type, bias, activation);
    else
        throw std::invalid_argument("quantized GEMM output must be F16 or BF16");
}
void fp8_hopper_gemm(void *x, const float *sx, void *w, const float *sw, void *y, int rows, int n,
                     int k, void *workspace, size_t bytes, cudaStream_t stream, int channels,
                     int tokens, int splits, Type output_type) {
    typed_hopper_gemm<cutlass::float_e4m3_t>(x, sx, w, sw, y, rows, n, k, workspace, bytes, stream,
                                             channels, tokens, splits, output_type);
}
void int8_hopper_gemm(void *x, const float *sx, void *w, const float *sw, void *y, int rows, int n,
                      int k, void *workspace, size_t bytes, cudaStream_t stream, int channels,
                      int tokens, int splits, Type output_type) {
    typed_hopper_gemm<int8_t>(x, sx, w, sw, y, rows, n, k, workspace, bytes, stream, channels,
                              tokens, splits, output_type);
}

void int8_hopper_finish(void *x, const float *sx, void *w, const float *sw, void *y, int rows,
                        int n, int k, void *workspace, size_t bytes, cudaStream_t stream,
                        int channels, int tokens, Type output_type, const void *bias,
                        ops::Activation activation) {
    if (activation != ops::Activation::Identity && activation != ops::Activation::Relu &&
        activation != ops::Activation::GeluTanh)
        throw std::invalid_argument("unsupported native Finish activation");
    typed_hopper_gemm<int8_t, native_direct::FinishPolicy>(x, sx, w, sw, y, rows, n, k,
        workspace, bytes, stream, channels, tokens, 1, output_type, bias, activation);
}
KernelResources int8_hopper_finish_resources(int channels, int tokens, Type type) {
    KernelResources result;
    auto inspect = [&](auto kernel) { result = decltype(kernel)::resources(); };
    if (type == Type::F16)
        native_direct::visit<int8_t, cutlass::half_t, native_direct::FinishPolicy>(channels, tokens, inspect);
    else if (type == Type::BF16)
        native_direct::visit<int8_t, cutlass::bfloat16_t, native_direct::FinishPolicy>(channels, tokens, inspect);
    else throw std::invalid_argument("quantized GEMM output must be F16 or BF16");
    return result;
}
} // namespace tts
