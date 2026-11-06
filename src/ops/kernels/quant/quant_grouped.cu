#include "ops/kernels/quant/quant_dispatch.h"
#include <cutlass/arch/mma_sm80.h>
#include <cutlass/arch/mma_sm89.h>
#include <cutlass/arch/memory_sm80.h>
#include <cutlass/arch/memory_sm75.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>
#include "ops/kernels/quant/quant_reduce.cuh"

namespace tts {
namespace {
// K groups are reduced in the native accumulator before scaling into FP32.
// Four warps share an M16/N64/K128 stage. Padding removes the eight-way bank
// conflicts of direct register-fragment loads from row-major byte matrices.
template <Type Format, class Output, int GroupSize, int Tokens>
__global__
__launch_bounds__(128) void grouped_gemm(const unsigned char *input, const float *input_scales,
                                         const unsigned char *weights, const float *weight_scales,
                                         Output *output, KernelProblem problem, int splits) {
    using Element = std::conditional_t<Format == Type::I8, int8_t, cutlass::float_e4m3_t>;
    using Accumulator = std::conditional_t<Format == Type::I8, int, float>;
    using Operator = std::conditional_t<Format == Type::I8, cutlass::arch::OpMultiplyAddSaturate,
                                        cutlass::arch::OpMultiplyAdd>;
    using Mma = cutlass::arch::Mma<cutlass::gemm::GemmShape<16, 8, 32>, 32, Element,
                                   cutlass::layout::RowMajor, Element, cutlass::layout::ColumnMajor,
                                   Accumulator, cutlass::layout::RowMajor, Operator>;
    constexpr int stride = 144, stage_k = 128;
    __shared__ unsigned char stages[2][(Tokens + 64) * stride];
    __shared__ float scale_stages[2][4 * (Tokens + 64)];
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    int row_base = blockIdx.y * Tokens, column_base = blockIdx.x * 64;
    int input_shift = __ffs(problem.input_group) - 1,
        weight_shift = __ffs(problem.weight_group) - 1;
    float result[Tokens / 16][2][4] = {};
    auto stage_operands = [&](int tile, int stage) {
        auto stage_rows = [&](const unsigned char *source, unsigned char *target, int first_row,
                              int limit, int row_count) {
#pragma unroll
            for (int part = 0; part < row_count / 16; ++part) {
                int row = threadIdx.x / 8 + part * 16, column = threadIdx.x % 8 * 16;
                bool valid = first_row + row < limit && tile + column < problem.reduction;
                auto address =
                    source +
                    (valid ? size_t(first_row + row) * problem.reduction + tile + column : 0);
                cutlass::arch::cp_async_zfill<16, cutlass::arch::CacheOperation::Global>(
                    target + row * stride + column, address, valid);
            }
        };
        stage_rows(input, stages[stage], row_base, problem.rows, Tokens);
        stage_rows(weights, stages[stage] + Tokens * stride, column_base, problem.channels, 64);
        constexpr int groups = stage_k / GroupSize;
        if (problem.input_group)
            for (int index = threadIdx.x; index < groups * Tokens; index += 128) {
                int row = row_base + index % Tokens, column = tile + index / Tokens * GroupSize;
                bool valid = row < problem.rows && column < problem.reduction;
                auto address =
                    input_scales + (valid ? size_t(column >> input_shift) * problem.rows + row : 0);
                cutlass::arch::cp_async_zfill<4, cutlass::arch::CacheOperation::Always>(
                    scale_stages[stage] + index, address, valid);
            }
        if (problem.weight_group)
            for (int index = threadIdx.x; index < groups * 64; index += 128) {
                int row = column_base + index % 64, column = tile + index / 64 * GroupSize;
                bool valid = row < problem.channels && column < problem.reduction;
                auto address =
                    weight_scales +
                    (valid ? size_t(column >> weight_shift) * problem.channels + row : 0);
                cutlass::arch::cp_async_zfill<4, cutlass::arch::CacheOperation::Always>(
                    scale_stages[stage] + 4 * Tokens + index, address, valid);
            }
        cutlass::arch::cp_async_fence();
    };
    int first_k = blockIdx.z * (problem.reduction / splits);
    int last_k = first_k + problem.reduction / splits;
    stage_operands(first_k, 0);
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();
    for (int tile = first_k, stage = 0; tile < last_k; tile += stage_k, stage ^= 1) {
        if (tile + stage_k < last_k)
            stage_operands(tile + stage_k, stage ^ 1);
        const auto *staged_input = stages[stage];
        const auto *staged_weight = staged_input + Tokens * stride;
#pragma unroll
        for (int group_index = 0; group_index < stage_k / GroupSize; ++group_index) {
            constexpr int steps = GroupSize / 32;
            int group = group_index * GroupSize;
            typename Mma::FragmentC partial[Tokens / 16][2]{};
#pragma unroll
            for (int iteration = 0; iteration < steps; ++iteration) {
                int step = iteration * 32;
                typename Mma::FragmentB weight[2];
#pragma unroll
                for (int fragment = 0; fragment < 2; ++fragment)
                    cutlass::arch::ldsm<cutlass::layout::RowMajor, 2>(
                        *reinterpret_cast<cutlass::Array<unsigned, 2> *>(&weight[fragment]),
                        staged_weight + (warp * 16 + fragment * 8 + lane % 8) * stride + group +
                            step + lane / 8 % 2 * 16);
#pragma unroll
                for (int token = 0; token < Tokens / 16; ++token) {
                    typename Mma::FragmentA activation;
                    cutlass::arch::ldsm<cutlass::layout::RowMajor, 4>(
                        *reinterpret_cast<cutlass::Array<unsigned, 4> *>(&activation),
                        staged_input + (token * 16 + lane % 16) * stride + group + step +
                            lane / 16 * 16);
#pragma unroll
                    for (int fragment = 0; fragment < 2; ++fragment)
                        Mma{}(partial[token][fragment], activation, weight[fragment],
                              partial[token][fragment]);
                }
            }
            float weight_scale[2][2] = {{1.f, 1.f}, {1.f, 1.f}};
            if (problem.weight_group) {
#pragma unroll
                for (int fragment = 0; fragment < 2; ++fragment)
#pragma unroll
                    for (int pair = 0; pair < 2; ++pair)
                        weight_scale[fragment][pair] =
                            scale_stages[stage][4 * Tokens + group_index * 64 + warp * 16 +
                                                fragment * 8 + lane % 4 * 2 + pair];
            }
#pragma unroll
            for (int token = 0; token < Tokens / 16; ++token) {
                float activation_scale[2] = {1.f, 1.f};
                if (problem.input_group)
#pragma unroll
                    for (int half = 0; half < 2; ++half)
                        activation_scale[half] =
                            scale_stages[stage]
                                        [group_index * Tokens + token * 16 + lane / 4 + half * 8];
#pragma unroll
                for (int fragment = 0; fragment < 2; ++fragment) {
#pragma unroll
                    for (int item = 0; item < 4; ++item) {
                        float scale = activation_scale[item / 2] * weight_scale[fragment][item % 2];
                        float value;
                        if constexpr (Format == Type::I8) {
                            // K128 signed-byte sums fit the exact ALU/FADD conversion range.
                            value = __int_as_float(partial[token][fragment][item] + 0x4b400000) -
                                    12582912.f;
                        } else {
                            value = partial[token][fragment][item];
                        }
                        result[token][fragment][item] += value * scale;
                    }
                }
            }
        }
        cutlass::arch::cp_async_wait<0>();
        __syncthreads();
    }
#pragma unroll
    for (int token = 0; token < Tokens / 16; ++token)
#pragma unroll
        for (int fragment = 0; fragment < 2; ++fragment) {
#pragma unroll
            for (int item = 0; item < 4; ++item) {
                int row = row_base + token * 16 + lane / 4 + item / 2 * 8;
                int column = column_base + warp * 16 + fragment * 8 + lane % 4 * 2 + item % 2;
                if (row < problem.rows && column < problem.channels) {
                    float scale = problem.input_group ? 1.f : input_scales[row];
                    if (!problem.weight_group)
                        scale *= weight_scales[column];
                    output[size_t(blockIdx.z) * problem.rows * problem.channels + size_t(row) * problem.channels + column] =
                        Output(result[token][fragment][item] * scale);
                }
            }
        }
}
} // namespace
static void launch_grouped_split(KernelProblem problem, const void *input, const float *input_scales,
                        const void *weights, const float *weight_scales, void *output,
                        cudaStream_t stream, Type format, Type output_type, int tokens, int splits) {
    auto launch = [&](auto value, auto storage, auto group, auto token_count) {
        using Output = decltype(value);
        grouped_gemm<decltype(storage)::value, Output, decltype(group)::value,
                     decltype(token_count)::value>
            <<<dim3((problem.channels + 63) / 64,
                    (problem.rows + decltype(token_count)::value - 1) /
                        decltype(token_count)::value, splits),
               128, 0, stream>>>(static_cast<const unsigned char *>(input), input_scales,
                                 static_cast<const unsigned char *>(weights), weight_scales,
                                 static_cast<Output *>(output), problem, splits);
    };
    int group_size = problem.input_group ? problem.input_group : problem.weight_group;
    if (problem.weight_group)
        group_size = std::min(group_size, problem.weight_group);
    auto dispatch_format = [&](auto value, auto group, auto token_count) {
        if (format == Type::I8)
            launch(value, std::integral_constant<Type, Type::I8>{}, group, token_count);
        else
            launch(value, std::integral_constant<Type, Type::FP8>{}, group, token_count);
    };
    auto dispatch_group = [&](auto value, auto token_count) {
        if (group_size == 32)
            dispatch_format(value, std::integral_constant<int, 32>{}, token_count);
        else if (group_size == 64)
            dispatch_format(value, std::integral_constant<int, 64>{}, token_count);
        else if (group_size == 128)
            dispatch_format(value, std::integral_constant<int, 128>{}, token_count);
        else
            throw std::invalid_argument("unsupported GEMM quantization group size");
    };
    auto dispatch = [&](auto value) {
        if (tokens == 16)
            dispatch_group(value, std::integral_constant<int, 16>{});
        else if (tokens == 32)
            dispatch_group(value, std::integral_constant<int, 32>{});
        else if (tokens == 64)
            dispatch_group(value, std::integral_constant<int, 64>{});
        else
            throw std::invalid_argument("unsupported grouped GEMM token tile");
    };
    if (output_type == Type::F32)
        dispatch(float{});
    else if (output_type == Type::F16)
        dispatch(__half{});
    else
        dispatch(__nv_bfloat16{});
}
void grouped_quant_gemm(KernelProblem problem, const void *input, const float *input_scales,
                        const void *weights, const float *weight_scales, void *output,
                        cudaStream_t stream, Type format, Type output_type, int tokens,
                        int splits, void *workspace, size_t workspace_bytes) {
    if (splits < 1 || (splits > 1 && problem.reduction % (128 * splits)))
        throw std::invalid_argument("grouped GEMM requires whole K128 partitions");
    if (splits == 1) {
        launch_grouped_split(problem, input, input_scales, weights, weight_scales,
                             output, stream, format, output_type, tokens, 1);
        return;
    }
    if (!workspace || size_t(problem.rows) * problem.channels * splits * sizeof(float) > workspace_bytes)
        throw std::invalid_argument("grouped GEMM partial workspace overflow");
    // Each partition applies its own scales before storing FP32 partials.
    launch_grouped_split(problem, input, input_scales, weights, weight_scales,
                         workspace, stream, format, Type::F32, tokens, splits);
    launch_quant_split_reduce(static_cast<const float *>(workspace), nullptr, nullptr,
                              output, problem.rows, problem.channels, splits, stream, output_type);
}
} // namespace tts
