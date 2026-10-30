#include "kernels/quant/quant_gemm.h"
#include "kernels/quant/quant_reduce.cuh"
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
                    int k, void *workspace, size_t bytes, cudaStream_t stream, int splits) {
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
        quant_split_reduce_scale<<<(m * n + 255) / 256, 256, 0, stream>>>(
            static_cast<Acc *>(workspace), sx, sw, static_cast<__nv_bfloat16 *>(y), m, n, splits);
    }
};
} // namespace native_partition

namespace native_direct {
using namespace cute;
using D = cutlass::bfloat16_t;
namespace fusion = cutlass::epilogue::fusion;
template <class E, int BM = 64, int BN = 16> struct Native {
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
    using EVT = fusion::Sm90EVT<MulOut, ScaleW, EVT0>;
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
    static void run(void *x, const float *sx, void *w, const float *sw, void *y, int m, int n,
                    int k, void *workspace, size_t bytes, cudaStream_t stream) {
        typename Mainloop::Arguments mainloop{};
        if constexpr (std::is_same_v<Acc, float>)
            mainloop.mma_promotion_interval = 4;
        mainloop.ptr_A = static_cast<E *>(w);
        mainloop.dA = {k, Int<1>{}, n * k};
        mainloop.ptr_B = static_cast<E *>(x);
        mainloop.dB = {k, Int<1>{}, m * k};
        typename Epilogue::Arguments epilogue{};
        epilogue.thread = {{sw}, {{sx}, {}, {}}, {}};
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
} // namespace native_direct

} // namespace tts

namespace tts {
template <class E>
void hopper_gemm(void *x, const float *sx, void *w, const float *sw, void *y, int rows, int n,
                 int k, void *workspace, size_t bytes, cudaStream_t stream, int tile_channels,
                 int tile_tokens, int splits) {
#define RUN_DIRECT(M, N)                                                                           \
    native_direct::Native<E, M, N>::run(x, sx, w, sw, y, rows, n, k, workspace, bytes, stream)
#define RUN_SPLIT(N)                                                                               \
    native_partition::Native<E, 64, N>::run(x, sx, w, sw, y, rows, n, k, workspace, bytes, stream, \
                                            splits)
    if (splits == 1) {
        if (tile_channels == 64 && tile_tokens == 16) {
            RUN_DIRECT(64, 16);
        } else if (tile_channels == 128 && tile_tokens == 16) {
            RUN_DIRECT(128, 16);
        } else if (tile_channels == 64 && tile_tokens == 32) {
            RUN_DIRECT(64, 32);
        } else if (tile_channels == 128 && tile_tokens == 32) {
            RUN_DIRECT(128, 32);
        } else if (tile_channels == 64 && tile_tokens == 64) {
            RUN_DIRECT(64, 64);
        } else if (tile_channels == 128 && tile_tokens == 64) {
            RUN_DIRECT(128, 64);
        } else if (tile_channels == 64 && tile_tokens == 128) {
            RUN_DIRECT(64, 128);
        } else if (tile_channels == 128 && tile_tokens == 128) {
            RUN_DIRECT(128, 128);
        } else
            throw std::invalid_argument("invalid quantized tile");
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
#undef RUN_DIRECT
#undef RUN_SPLIT
}
void fp8_hopper_gemm(void *x, const float *sx, void *w, const float *sw, void *y, int rows, int n,
                     int k, void *workspace, size_t bytes, cudaStream_t stream, int channels,
                     int tokens, int splits) {
    hopper_gemm<cutlass::float_e4m3_t>(x, sx, w, sw, y, rows, n, k, workspace, bytes, stream,
                                       channels, tokens, splits);
}
void int8_hopper_gemm(void *x, const float *sx, void *w, const float *sw, void *y, int rows, int n,
                      int k, void *workspace, size_t bytes, cudaStream_t stream, int channels,
                      int tokens, int splits) {
    hopper_gemm<int8_t>(x, sx, w, sw, y, rows, n, k, workspace, bytes, stream, channels, tokens,
                        splits);
}
} // namespace tts
