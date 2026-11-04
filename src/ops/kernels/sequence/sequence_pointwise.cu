#include "ops/kernels/common/storage_dispatch.cuh"
#include "runtime/cuda_context.h"
#include "ops/kernels/sequence/sequence_internal.cuh"
#include <cstring>

namespace tts::ops {
template <class T, class Fusion>
__global__ void pointwise_kernel(View x, View y, int rows, int cols, Fusion f, int repeat) {
    size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= size_t(rows) * cols)
        return;
    int row = index / cols, col = index % cols;
    f.template store<T>(y, row, col, f.template apply<T>(x.get<T>(row / repeat, col), row, col));
}
template <class T, class Fusion>
__global__ void pointwise_row_kernel(View x, View y, int rows, int cols, Fusion f,
                                     cutlass::FastDivmod repeat, cutlass::FastDivmod sequence,
                                     RowTiles tiles) {
    int lanes = 1 << tiles.row_shift;
    int row_in_tile = threadIdx.x >> tiles.row_shift, lane = threadIdx.x & (lanes - 1);
    for (uint64_t tile = blockIdx.x; tile < tiles.count; tile += gridDim.x) {
        int row_base, col_base;
        tiles.coordinates(tile, row_base, col_base);
        if (row_in_tile >= rows - row_base)
            continue;
        int row = row_base + row_in_tile, input_row = repeat.div(row);
        auto prepared = f.prepare_row(row, sequence.div(row));
        float input[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int within = lane + j * lanes;
            if (within < cols - col_base)
                input[j] = x.get<T>(input_row, col_base + within);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int within = lane + j * lanes;
            if (within < cols - col_base) {
                int col = col_base + within;
                f.template store<T>(y, row, col, f.template apply<T>(input[j], row, col, &prepared));
            }
        }
    }
}
template <class T, bool Depthwise, class Fusion>
__global__ void conv_kernel(View x, View weight, View y, int rows, int input_length,
                            int output_length, ConvGeometry g, Fusion f, SequenceLayout input, SequenceLayout output) {
    size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= size_t(rows) * g.output_channels)
        return;
    int row = index / g.output_channels, col = index % g.output_channels;
    int b = row / output_length, time = row % output_length;
    int first_row = b * input_length, extent = input_length;
    if constexpr (std::is_same_v<Fusion, PackedFinish>) {
        auto address = sequence_address(output, row, output_length);
        b = address.sequence; time = address.position;
        first_row = input.cu_seqlens ? input.cu_seqlens[b] : b * input_length;
        extent = input.cu_seqlens ? input.cu_seqlens[b + 1] - first_row : input_length;
    }
    int per_group = g.output_channels / g.groups, group = col / per_group;
    float sum = 0;
    for (int tap = 0; tap < g.kernel; ++tap) {
        int source = g.transpose ? time + g.padding - tap * g.dilation
                                 : time * g.stride - g.padding + tap * g.dilation;
        if (g.transpose) {
            if (source % g.stride)
                continue;
            source /= g.stride;
        }
        if (source < 0 || source >= extent)
            continue;
        if constexpr (Depthwise) {
            int inputs = g.input_channels / g.groups;
            for (int c = 0; c < inputs; ++c)
                sum += x.get<T>(first_row + source, group * inputs + c) *
                       weight.get<T>((group * g.kernel + tap) * per_group + col % per_group, c);
        } else
            sum += x.get<float>(first_row + source,
                                (group * g.kernel + tap) * per_group + col % per_group);
    }
    f.template store<T>(y, row, col, f.template apply<T>(sum, row, col));
}
// A projected tap is a contiguous channel packet. Reuse its row address
// across four outputs and advance directly between valid taps.
template <class T, class Fusion>
__global__ void
conv_finish_row_kernel(View x, View y, int rows, int input_length, ConvGeometry g, Fusion f,
                       RowTiles tiles, cutlass::FastDivmod time_div, cutlass::FastDivmod group_div,
                       cutlass::FastDivmod dilation_div, cutlass::FastDivmod sequence_div,
                       SequenceLayout input, SequenceLayout output) {
    int lanes = 1 << tiles.row_shift;
    int row_in_tile = threadIdx.x >> tiles.row_shift, lane = threadIdx.x & (lanes - 1);
    for (uint64_t tile = blockIdx.x; tile < tiles.count; tile += gridDim.x) {
        int row_base, col_base;
        tiles.coordinates(tile, row_base, col_base);
        if (row_in_tile >= rows - row_base)
            continue;
        int row = row_base + row_in_tile, time;
        int batch = time_div.divmod(time, row);
        int first_row = batch * input_length, extent = input_length;
        if constexpr (std::is_same_v<Fusion, PackedFinish>) {
            if (output.row_batch) { batch = output.row_batch[row]; time = output.row_position[row]; }
            first_row = input.cu_seqlens ? input.cu_seqlens[batch] : batch * input_length;
            extent = input.cu_seqlens ? input.cu_seqlens[batch + 1] - first_row : input_length;
        }
        int position = g.transpose ? time + g.padding : time * g.stride - g.padding;
        int first = 0, last = g.kernel;
        if (g.transpose) {
            if (position >= extent)
                first = 1 + dilation_div.div(position - extent);
            last = position < 0 ? 0 : min(last, 1 + dilation_div.div(position));
        } else {
            if (position < 0)
                first = 1 + dilation_div.div(-1 - position);
            last = position >= extent
                       ? 0
                       : min(last, 1 + dilation_div.div(extent - 1 - position));
        }
        int offsets[4];
        float sums[4] = {};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int col = col_base + lane + j * lanes;
            offsets[j] = col + group_div.div(col) * (g.kernel - 1) * group_div.divisor;
        }
        if (first < last) {
            int direction = g.transpose ? -g.dilation : g.dilation;
            int source = position + first * direction;
            const float *tap = static_cast<const float *>(x.data) +
                               (size_t(first_row) + source) * x.stride +
                               size_t(first) * group_div.divisor;
            ptrdiff_t step = ptrdiff_t(direction) * x.stride + group_div.divisor;
            for (int k = first; k < last; ++k, tap += step) {
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    sums[j] += tap[offsets[j]];
            }
        }
        auto prepared = f.prepare_row(row, sequence_div.div(row));
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int col = col_base + lane + j * lanes;
            f.template store<T>(y, row, col, f.template apply<T>(sums[j], row, col, &prepared));
        }
    }
}
template <class T, Type Storage>
__global__ void pointwise_quantize_kernel(View x, View y, int cols, Finish f,
                                          cutlass::FastDivmod repeat, cutlass::FastDivmod sequence,
                                          QuantizedRowOutput<Storage> output, bool cached) {
    using Pack = quant_device::ValuePack<T, 4>;
    using Math = quant_device::Packed16<T>;
    using Pair = typename Math::Pair;
    __shared__ unsigned partial[32];
    extern __shared__ float storage[];
    auto *values = reinterpret_cast<T *>(storage);
    int row = blockIdx.x, input_row = repeat.div(row);
    const T *input =
        static_cast<const T *>(x.data) + size_t(x.rows == 1 ? 0 : input_row) * x.stride;
    T *retained = y.data ? static_cast<T *>(y.data) + size_t(row) * y.stride : nullptr;
    Finish::Row prepared{sequence.div(row), f.residual_rows ? f.residual_rows[row] : row};
    auto transform = [&](Pack input, int c, const Finish &finish) {
        Pack value{};
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            float2 v = Math::unpack(*reinterpret_cast<const Pair *>(input.v + p * 2));
            if (c + p * 2 < cols)
                value.v[p * 2] = T(finish.apply<T>(v.x, row, c + p * 2, &prepared));
            if (c + p * 2 + 1 < cols)
                value.v[p * 2 + 1] = T(finish.apply<T>(v.y, row, c + p * 2 + 1, &prepared));
        }
        return value;
    };
    constexpr int packets = 4;
    uint2 registers[packets]{};
    bool register_cached = cols <= int(blockDim.x) * 4 * packets;
    if (register_cached) {
        float amax = 0;
#pragma unroll
        for (int i = 0; i < packets; ++i) {
            int c = int(threadIdx.x) * 4 + i * int(blockDim.x) * 4;
            if (c >= cols)
                continue;
            Pack value = transform(quant_device::load_pack<Pack>(input + c, cols - c), c, f);
            memcpy(&registers[i], &value, sizeof(value));
            if (retained)
                quant_device::store_pack(retained + c, value, cols - c);
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                float2 v = Math::unpack(*reinterpret_cast<const Pair *>(value.v + p * 2));
                amax = fmaxf(amax, fmaxf(fabsf(v.x), fabsf(v.y)));
            }
        }
        unsigned bits = __reduce_max_sync(0xffffffffu, __float_as_uint(amax));
        if (!(threadIdx.x % 32))
            partial[threadIdx.x / 32] = bits;
        __syncthreads();
        bits = threadIdx.x % 32 < blockDim.x / 32 ? partial[threadIdx.x % 32] : 0u;
        amax = __uint_as_float(__reduce_max_sync(0xffffffffu, bits));
        float inverse = output.scale(row, amax);
#pragma unroll
        for (int i = 0; i < packets; ++i) {
            int c = int(threadIdx.x) * 4 + i * int(blockDim.x) * 4;
            if (c >= cols)
                continue;
            Pack value;
            memcpy(&value, &registers[i], sizeof(value));
            quant_device::BytePack<4> encoded;
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                float2 v = Math::unpack(*reinterpret_cast<const Pair *>(value.v + p * 2));
                encoded.v[p] =
                    quant_device::Format<Storage>::pair(make_float2(v.x * inverse, v.y * inverse));
            }
            quant_device::store_pack(output.values + size_t(row) * cols + c, encoded, cols - c);
        }
        return;
    }
    Finish first = f;
    if (!cached)
        first.raw.data = nullptr;
    float amax = 0;
    for (int c = int(threadIdx.x) * 4; c < cols; c += int(blockDim.x) * 4) {
        Pack value = transform(quant_device::load_pack<Pack>(input + c, cols - c), c, first);
        if (cached)
            quant_device::store_pack(values + c, value, cols - c);
        if (retained && cached)
            quant_device::store_pack(retained + c, value, cols - c);
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            float2 v = Math::unpack(*reinterpret_cast<const Pair *>(value.v + p * 2));
            amax = fmaxf(amax, fmaxf(fabsf(v.x), fabsf(v.y)));
        }
    }
    unsigned bits = __reduce_max_sync(0xffffffffu, __float_as_uint(amax));
    if (!(threadIdx.x % 32))
        partial[threadIdx.x / 32] = bits;
    __syncthreads();
    bits = threadIdx.x % 32 < blockDim.x / 32 ? partial[threadIdx.x % 32] : 0u;
    amax = __uint_as_float(__reduce_max_sync(0xffffffffu, bits));
    float inverse = output.scale(row, amax);
    for (int c = int(threadIdx.x) * 4; c < cols; c += int(blockDim.x) * 4) {
        Pack value = cached ? quant_device::load_pack<Pack>(values + c, cols - c)
                            : transform(quant_device::load_pack<Pack>(input + c, cols - c), c, f);
        if (!cached && retained)
            quant_device::store_pack(retained + c, value, cols - c);
        quant_device::BytePack<4> encoded;
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            float2 v = Math::unpack(*reinterpret_cast<const Pair *>(value.v + p * 2));
            encoded.v[p] =
                quant_device::Format<Storage>::pair(make_float2(v.x * inverse, v.y * inverse));
        }
        quant_device::store_pack(output.values + size_t(row) * cols + c, encoded, cols - c);
    }
}
int ConvGeometry::output_length(int input) const {
    if (input < 1 || input_channels < 1 || output_channels < 1 || kernel < 1 || stride < 1 ||
        padding < 0 || padding_right < -1 || dilation < 1 || groups < 1 ||
        input_channels % groups || output_channels % groups)
        throw std::invalid_argument("invalid convolution geometry");
    int64_t extent = int64_t(dilation) * (kernel - 1);
    int64_t padding_total = int64_t(padding) + (padding_right < 0 ? padding : padding_right);
    int64_t output = transpose ? int64_t(input - 1) * stride - padding_total + extent + 1
                               : (int64_t(input) + padding_total - extent - 1);
    if (!transpose)
        output = output < 0 ? 0 : output / stride + 1;
    if (output < 1 || output > INT32_MAX)
        throw std::invalid_argument("convolution output length out of range");
    return int(output);
}
void pointwise( TensorView x, TensorView y, const Epilogue &e) {
    const auto stream = execution_context(x).stream;
    check_output(y);
    check_epilogue(e, y);
    if (!x.data || e.input_repeat < 1 || x.type != y.type || x.cols != y.cols ||
        int64_t(x.rows) * e.input_repeat != y.rows || (e.input_repeat > 1 && x.data == y.data))
        throw std::invalid_argument("invalid pointwise input/repetition");
    RowTiles tiles(y.rows, y.cols);
    bool row_tiled = (e.gate.data || e.condition.data || e.raw.data || e.residual_rows ||
                      e.packed || e.condition_sources || e.output_rows) &&
                     tiles.full_rows(y.rows, y.cols);
    auto launch = [&](auto type, auto finish) {
        using T = decltype(type);
        if (row_tiled) {
            int grid = int(std::min<uint64_t>(tiles.count, INT32_MAX));
            pointwise_row_kernel<T, decltype(finish)><<<grid, 256, 0, stream>>>(
                View(x), View(y), y.rows, y.cols, finish, cutlass::FastDivmod(e.input_repeat),
                cutlass::FastDivmod(e.rows_per_sequence), tiles);
        } else {
            int grid = (y.numel() + 255) / 256;
            pointwise_kernel<T, decltype(finish)><<<grid, 256, 0, stream>>>(View(x), View(y), y.rows, y.cols,
                                                          finish, e.input_repeat);
        }
    };
    auto typed = [&](auto finish) {
        if (y.type == Type::BF16) launch(bf16{}, finish);
        else if (y.type == Type::F16) launch(__half{}, finish);
        else launch(float{}, finish);
    };
    if (e.packed || e.condition_sources || e.output_rows) typed(PackedFinish(e));
    else typed(Finish(e));
    CUDA(cudaGetLastError());
}
TensorShape validate_pointwise_quantize(TensorView x, RowScaledView q,
                                       const Epilogue &e, TensorView retained) {
    if (e.packed || e.condition_sources || e.output_rows)
        throw std::invalid_argument("quantized pointwise does not accept indirect epilogue addressing");
    auto shape = check_quantized_output(x, q, retained);
    check_epilogue(e, shape, x.type);
    if (!x.data || e.input_repeat < 1 || x.cols != shape.cols ||
        int64_t(x.rows) * e.input_repeat != shape.rows ||
        (e.input_repeat > 1 && x.data == retained.data))
        throw std::invalid_argument("invalid quantized pointwise input/repetition");
    check_quantized_aliases(q, {x, retained, e.bias, e.residual, e.raw, e.gate, e.condition});
    check_retained_raw(retained, e.raw, {e.bias, e.gate, e.condition});
    for (auto output : {retained, e.raw}) {
        check_row_alias(output, x, e.input_repeat == 1);
        check_row_alias(output, e.residual, !e.residual_rows);
    }
    return shape;
}
void pointwise_quantize( TensorView x, RowScaledView q,
                        const Epilogue &e, TensorView retained, int multiprocessors) {
    const auto stream = execution_context(x).stream;
    auto shape = validate_pointwise_quantize(x, q, e, retained);
    bool cached = size_t(shape.cols) * bytes(x.type) <= norm_cache_bytes;
    size_t shared = cached ? size_t(shape.cols) * bytes(x.type) : 0;
    int threads = 256;
    if (shape.rows < multiprocessors)
        while (threads < 1024 && threads * 4 < shape.cols) threads *= 2;
    if (shape.cols <= threads * 4 * 4) shared = 0;

    auto launch = [&](auto type, auto output) {
        pointwise_quantize_kernel<decltype(type), decltype(output)::storage_type>
            <<<shape.rows, threads, shared, stream>>>(
                View(x), View(retained), shape.cols, Finish(e), cutlass::FastDivmod(e.input_repeat),
                cutlass::FastDivmod(e.rows_per_sequence), output, cached);
    };
    if (x.type == Type::F16) {
        if (q.values.type == Type::I8) launch(__half{}, QuantizedRowOutput<Type::I8>(q));
        else launch(__half{}, QuantizedRowOutput<Type::FP8>(q));
    } else {
        if (q.values.type == Type::I8) launch(bf16{}, QuantizedRowOutput<Type::I8>(q));
        else launch(bf16{}, QuantizedRowOutput<Type::FP8>(q));
    }
    CUDA(cudaGetLastError());
}
template <class T>
__global__ void linear_resample_kernel(View x, View y, int rows, int cols, int input_time,
                                      int output_time, float scale, cutlass::FastDivmod sequence,
                                      RowTiles tiles, SequenceLayout input, SequenceLayout output) {
    int lanes = 1 << tiles.row_shift;
    int row_in_tile = threadIdx.x >> tiles.row_shift, lane = threadIdx.x & (lanes - 1);
    for (uint64_t tile = blockIdx.x; tile < tiles.count; tile += gridDim.x) {
        int row_base, col_base;
        tiles.coordinates(tile, row_base, col_base);
        if (row_in_tile >= rows - row_base) continue;
        int row = row_base + row_in_tile, batch = sequence.div(row);
        int first = batch * input_time, position_row = row - batch * output_time;
        int itime = input_time, otime = output_time;
        if (input.cu_seqlens || output.cu_seqlens) {
            auto address = sequence_address(output, row, output_time);
            batch = address.sequence; position_row = address.position;
            first = input.cu_seqlens ? input.cu_seqlens[batch] : batch * input_time;
            itime = input.cu_seqlens ? input.cu_seqlens[batch + 1] - first : input_time;
            otime = address.extent;
            scale = float(itime) / otime;
        }
        float position = fmaxf(scale * (position_row + .5f) - .5f, 0.f);
        int left = min(int(position), itime - 1), right = min(left + 1, itime - 1);
        float weight = position - left;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int column = col_base + lane + j * lanes;
            if (column < cols)
                y.set<T>(row, column, (input.cu_seqlens || output.cu_seqlens) && itime == otime
                    ? x.get<T>(first + position_row, column)
                    : (1.f - weight) * x.get<T>(first + left, column) +
                                          weight * x.get<T>(first + right, column));
        }
    }
}
void linear_resample( TensorView x, TensorView y, int batch,
                     int input_time, int output_time, const SequenceLayout *input,
                     const SequenceLayout *output) {
    const auto stream = execution_context(x).stream;
    check_output(y);
    if (batch < 1 || input_time < 1 || output_time < 1 ||
        x.rows != (input ? input->rows : int64_t(batch) * input_time) ||
        y.rows != (output ? output->rows : int64_t(batch) * output_time) || x.cols != y.cols || x.type != y.type ||
        !x.data || x.overlaps(y))
        throw std::invalid_argument("invalid temporal interpolation");
    RowTiles tiles(y.rows, y.cols);
    auto launch = [&](auto type) {
        int grid = int(std::min<uint64_t>(tiles.count, INT32_MAX));
        linear_resample_kernel<decltype(type)><<<grid, 256, 0, stream>>>(
            View(x), View(y), y.rows, y.cols, input_time, output_time, float(input_time) / output_time,
            cutlass::FastDivmod(output_time), tiles,
            input ? *input : SequenceLayout{}, output ? *output : SequenceLayout{});
    };
    if (y.type == Type::BF16) launch(bf16{});
    else if (y.type == Type::F16) launch(__half{});
    else launch(float{});
    CUDA(cudaGetLastError());
}
template <bool Depthwise>
static void convolution(cudaStream_t stream, TensorView x, TensorView weight, TensorView y,
                        int batch, int input_length, const ConvGeometry &g, const Epilogue &e,
                        const SequenceLayout *input, const SequenceLayout *output) {
    check_output(y);
    check_epilogue(e, y);
    int output_length = g.output_length(input_length);
    if (bool(input) != bool(output) || batch < 1 ||
        (output ? output->rows : int64_t(batch) * output_length) != y.rows || y.cols != g.output_channels ||
        (input ? input->rows : int64_t(batch) * input_length) != x.rows ||
        (Depthwise ? (x.type != y.type || x.cols != g.input_channels || weight.type != x.type ||
                      weight.rows != g.kernel * g.output_channels ||
                      weight.cols != g.input_channels / g.groups)
                   : (x.type != Type::F32 || x.cols != g.kernel * g.output_channels)))
        throw std::invalid_argument("convolution workspace mismatch");
    auto run = [&](auto finish) {
    if constexpr (!Depthwise) {
        RowTiles tiles(y.rows, y.cols);
        // Strided transpose retains the generic tap congruence handling.
        if ((!g.transpose || g.stride == 1) && y.rows >= tiles.rows_per_tile &&
            y.cols % tiles.columns_per_tile == 0) {
            auto launch = [&](auto type) {
                int grid = int(std::min<uint64_t>(tiles.count, INT32_MAX));
                conv_finish_row_kernel<decltype(type), decltype(finish)><<<grid, 256, 0, stream>>>(
                    View(x), View(y), y.rows, input_length, g, finish, tiles,
                    cutlass::FastDivmod(output_length),
                    cutlass::FastDivmod(g.output_channels / g.groups),
                    cutlass::FastDivmod(g.dilation), cutlass::FastDivmod(e.rows_per_sequence),
                    input ? *input : SequenceLayout{}, output ? *output : SequenceLayout{});
            };
            if (y.type == Type::BF16)
                launch(bf16{});
            else if (y.type == Type::F16)
                launch(__half{});
            else
                launch(float{});
            CUDA(cudaGetLastError());
            return;
        }
    }
    int grid = (y.numel() + 255) / 256;
    cuda_kernels::dispatch_storage(y.type, [&](auto storage) {
        conv_kernel<decltype(storage), Depthwise, decltype(finish)><<<grid, 256, 0, stream>>>(
            View(x), View(weight), View(y), y.rows, input_length, output_length, g, finish,
            input ? *input : SequenceLayout{}, output ? *output : SequenceLayout{});
    });
    CUDA(cudaGetLastError());
    };
    if (input || output || e.packed || e.condition_sources || e.output_rows) run(PackedFinish(e));
    else run(Finish(e));
}
void convolution_finish( TensorView p, TensorView y, int b, int t,
                        const ConvGeometry &g, const Epilogue &e,
                        const SequenceLayout *input, const SequenceLayout *output) {
    const auto stream = execution_context(p).stream;
    convolution<false>(stream, p, {}, y, b, t, g, e, input, output);
}
void direct_convolution( TensorView x, TensorView w, TensorView y, int b, int t,
                        const ConvGeometry &g, const Epilogue &e,
                        const SequenceLayout *input, const SequenceLayout *output) {
    const auto stream = execution_context(x).stream;
    convolution<true>(stream, x, w, y, b, t, g, e, input, output);
}
} // namespace tts::ops
