#include "ops/kernels/quant/quant_gemm.h"
#include "ops/kernels/quant/quant_reduce.cuh"
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#include <stdexcept>
#include <type_traits>

namespace tts {
namespace {
using namespace cute;
namespace ep = cutlass::epilogue::threadblock;
template <int Tokens, int Channels, class D> struct Sm80Gemm {
    static constexpr bool Split = std::is_same_v<D, int32_t>;
    using Tile = cutlass::gemm::GemmShape<Tokens, Channels, Tokens == 128 ? 64 : 128>;
    using Warp = cutlass::gemm::GemmShape<(Tokens < 64 ? Tokens : 64), 64, 64>;
    using Map = ep::OutputTileThreadLayout<Tile, Warp, float, 4, 1>;
    using Out = D;
    using Store = ep::VisitorAuxStore<Map, Out, cutlass::FloatRoundStyle::round_to_nearest,
                                      Stride<int64_t, _1, int64_t>>;
    using ScaleX = ep::VisitorColBroadcast<Map, float, Stride<_1, _0, _0>, false>;
    using ScaleW = ep::VisitorRowBroadcast<Map, float, Stride<_0, _1, _0>, false>;
    using Mul = ep::VisitorCompute<cutlass::multiplies, float, float,
                                   cutlass::FloatRoundStyle::round_to_nearest>;
    using MulOut = ep::VisitorCompute<cutlass::multiplies, D, float,
                                      cutlass::FloatRoundStyle::round_to_nearest>;
    using Scaled = ep::Sm80EVT<MulOut, ScaleW, ep::Sm80EVT<Mul, ScaleX, ep::VisitorAccFetch>>;
    using Visitor = ep::Sm80EVT<Store, std::conditional_t<Split, ep::VisitorAccFetch, Scaled>>;
    using Kernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        int8_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 16, int8_t,
        cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 16, float,
        cutlass::layout::RowMajor, 4, int32_t, float, cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80, Tile, Warp, cutlass::gemm::GemmShape<16, 8, 32>, Visitor,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3,
        cutlass::arch::OpMultiplyAddSaturate>::GemmKernel;
    // Ampere/Ada expose at least 96 KiB of opt-in shared memory per CTA.
    static_assert(sizeof(typename Kernel::SharedStorage) <= 96 * 1024);
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
    static void run(void *x, const float *sx, void *w, const float *sw, void *y, int m, int n,
                    int k, void *workspace, size_t bytes, cudaStream_t stream, int splits,
                    Type output_type) {
        typename Visitor::Arguments epilogue{};
        if constexpr (Split) {
            if (size_t(m) * n * splits * sizeof(int32_t) > bytes)
                throw std::invalid_argument("INT8 partial workspace overflow");
            epilogue = {{}, {static_cast<int32_t *>(workspace), {n, _1{}, int64_t(m) * n}}};
        } else {
            epilogue = {{{sw}, {{sx}, {}, {}}, {}}, {static_cast<Out *>(y), {n, _1{}, 0}}};
        }
        typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kBatched,
                                      {m, n, k / splits},
                                      splits,
                                      epilogue,
                                      x,
                                      w,
                                      nullptr,
                                      nullptr,
                                      k / splits,
                                      k / splits,
                                      0,
                                      0,
                                      k,
                                      k,
                                      n,
                                      n};
        auto check = [](cutlass::Status status) {
            if (status != cutlass::Status::kSuccess)
                throw std::runtime_error(cutlassGetStatusString(status));
        };
        Gemm gemm;
        check(gemm.can_implement(args));
        // Batched K views and EVT need no library scratch, memset or repacking.
        if (gemm.get_workspace_size(args))
            throw std::logic_error("unexpected SM80 GEMM workspace");
        check(gemm.initialize(args, nullptr, stream));
        check(gemm.run(stream));
        if constexpr (Split)
            launch_quant_split_reduce(static_cast<const int32_t *>(workspace), sx, sw, y, m, n,
                                      splits, stream, output_type);
    }
};
} // namespace
template <class D>
void sm80_gemm(void *x, const float *sx, void *w, const float *sw, void *y, int m, int n, int k,
               void *workspace, size_t bytes, cudaStream_t stream, int channels, int tokens,
               int splits, Type output_type) {
    if (splits < 1 || k % splits || (k / splits) % 16)
        throw std::invalid_argument("unaligned INT8 K partition");
#define RUN(M, N)                                                                                  \
    if (splits == 1)                                                                               \
        Sm80Gemm<M, N, D>::run(x, sx, w, sw, y, m, n, k, workspace, bytes, stream, splits,         \
                               output_type);                                                       \
    else                                                                                           \
        Sm80Gemm<M, N, int32_t>::run(x, sx, w, sw, y, m, n, k, workspace, bytes, stream, splits,   \
                                     output_type)
#define TILE(M)                                                                                    \
    case M:                                                                                        \
        if (channels == 64) {                                                                      \
            RUN(M, 64);                                                                            \
        } else if (channels == 128) {                                                              \
            RUN(M, 128);                                                                           \
        } else                                                                                     \
            throw std::invalid_argument("invalid SM80 channel tile");                              \
        break
    switch (tokens) {
        TILE(16);
        TILE(32);
        TILE(64);
        TILE(128);
    default:
        throw std::invalid_argument("invalid SM80 token tile");
    }
#undef TILE
#undef RUN
}
void int8_sm80_gemm(void *x, const float *sx, void *w, const float *sw, void *y, int m, int n,
                    int k, void *workspace, size_t bytes, cudaStream_t stream, int channels,
                    int tokens, int splits, Type output_type) {
    if (output_type == Type::F16)
        sm80_gemm<cutlass::half_t>(x, sx, w, sw, y, m, n, k, workspace, bytes, stream, channels,
                                   tokens, splits, output_type);
    else if (output_type == Type::BF16)
        sm80_gemm<cutlass::bfloat16_t>(x, sx, w, sw, y, m, n, k, workspace, bytes, stream, channels,
                                       tokens, splits, output_type);
    else
        throw std::invalid_argument("quantized GEMM output must be F16 or BF16");
}
} // namespace tts
