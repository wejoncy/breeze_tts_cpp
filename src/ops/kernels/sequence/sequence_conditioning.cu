#include "ops/cuda/backend.h"
#include "ops/kernels/sequence/sequence_internal.cuh"
#include <curand_kernel.h>

namespace tts::ops {
template <class T>
__global__ void position_kernel(View x, View y, View table, View alpha, int rows, int cols,
                                int time, int offset, const void *const *sources, void *const *retained) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(rows) * cols)
        return;
    int row = i / cols, c = i % cols;
    float scale = alpha.data ? alpha.get<T>(0, 0) : 1.f;
    float value = sources ? (sources[row] ? float(static_cast<const T *>(sources[row])[c]) : 0.f)
                          : x.get<T>(row, c);
    if (retained && retained[row]) static_cast<T *>(retained[row])[c] = T(value);
    y.set<T>(row, c, value + scale * table.get<T>(row % time + offset, c));
}
void positional_add( TensorView x, TensorView y, TensorView table, int time,
                    int offset, TensorView alpha, const void *const *sources,
                    void *const *retained) {
    const auto stream = execution_context(y, {x, table, alpha}).stream;
    check_output(y);
    if (!sources)
        check_same(x, y);
    if (time < 1 || y.rows % time || offset < 0 || time > table.rows - offset ||
        table.cols != y.cols || table.type != y.type ||
        (alpha.data && (alpha.numel() != 1 || alpha.type != y.type)))
        throw std::invalid_argument("invalid positional embedding");
    int blocks = (y.numel() + 255) / 256;
    if (y.type == Type::BF16)
        position_kernel<bf16><<<blocks, 256, 0, stream>>>(View(x), View(y), View(table), View(alpha),
                                                     y.rows, y.cols, time, offset, sources, retained);
    else if (y.type == Type::F16)
        position_kernel<__half><<<blocks, 256, 0, stream>>>(View(x), View(y), View(table), View(alpha),
                                                       y.rows, y.cols, time, offset, sources, retained);
    else
        position_kernel<float><<<blocks, 256, 0, stream>>>(View(x), View(y), View(table), View(alpha),
                                                      y.rows, y.cols, time, offset, sources, retained);
    CUDA(cudaGetLastError());
}
template <class T> __device__ __forceinline__ void rotate(float &a, float &b, float co, float si) {
    float first = float(T(a * co)) - float(T(b * si));
    b = float(T(a * si)) + float(T(b * co));
    a = first;
}
template <class T>
__global__ void rotary_flat_kernel(View x, View bias, View rotary, int rows, int width, int time,
                                   int head_dim, int dim, bool per_head, int offset) {
    size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= size_t(rows) * width * 3 / 2)
        return;
    // Each thread owns a pair, so in-place rotation never races its neighbor.
    int pair_width = width / 2;
    int pair = index % pair_width, part = (index / pair_width) % 3;
    int row = index / (pair_width * 3), c = pair * 2;
    int local = per_head ? c % head_dim : c;
    int base = part * width + c;
    float a = x.get<T>(row, base) + bias.get<T>(0, base);
    float b = x.get<T>(row, base + 1) + bias.get<T>(0, base + 1);
    // Bias projection output is a model-dtype boundary before rotation.
    a = float(T(a));
    b = float(T(b));
    if (part < 2 && local < dim) {
        int pos = row % time + offset;
        float co = rotary.get<T>(pos, local / 2), si = rotary.get<T>(pos, dim / 2 + local / 2);
        rotate<T>(a, b, co, si);
    }
    x.set<T>(row, base, a);
    x.set<T>(row, base + 1, b);
}
template <class T>
__global__ void rotary_row_kernel(View x, View bias, View rotary, int rows, int width, int time,
                                  int head_dim, int dim, bool per_head, int offset, RowTiles tiles,
                                  cutlass::FastDivmod head_channels, cutlass::FastDivmod sequence) {
    int lanes = 1 << tiles.row_shift;
    int row_in_tile = threadIdx.x >> tiles.row_shift, lane = threadIdx.x & (lanes - 1);
    for (uint64_t tile = blockIdx.x; tile < tiles.count; tile += gridDim.x) {
        int row_base, pair_base;
        tiles.coordinates(tile, row_base, pair_base);
        if (row_in_tile >= rows - row_base)
            continue;
        int row = row_base + row_in_tile, pos = sequence.rem(row) + offset;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int within = lane + j * lanes;
            if (within >= width / 2 - pair_base)
                continue;
            int c = (pair_base + within) * 2;
            int local = per_head ? head_channels.rem(c) : c;
            float co = 0, si = 0;
            if (local < dim) {
                co = rotary.get<T>(pos, local / 2);
                si = rotary.get<T>(pos, dim / 2 + local / 2);
            }
#pragma unroll
            for (int part = 0; part < 3; ++part) {
                int base = part * width + c;
                float a = float(T(x.get<T>(row, base) + bias.get<T>(0, base)));
                float b = float(T(x.get<T>(row, base + 1) + bias.get<T>(0, base + 1)));
                if (part < 2 && local < dim) {
                    rotate<T>(a, b, co, si);
                }
                x.set<T>(row, base, a);
                x.set<T>(row, base + 1, b);
            }
        }
    }
}
void qkv_rotary( TensorView x, TensorView bias, TensorView rotary, int time,
                int heads, int head_dim, int dim, bool per_head, int offset) {
    const auto stream = execution_context(x).stream;
    check_output(x);
    check_same(bias, x, true);
    int64_t width = int64_t(heads) * head_dim;
    if (heads < 1 || head_dim < 1 || head_dim % 2 || width * 3 != x.cols || time < 1 ||
        x.rows % time || offset < 0 || dim < 0 || dim % 2 || dim > (per_head ? head_dim : width) ||
        (dim && (!rotary.data || rotary.type != x.type || rotary.cols != dim ||
                 time > rotary.rows - offset)))
        throw std::invalid_argument("invalid rotary projection");
    RowTiles tiles(x.rows, int(width / 2));
    auto launch = [&](auto type) {
        using T = decltype(type);
        if (tiles.full_rows(x.rows, int(width / 2))) {
            int grid = int(std::min<uint64_t>(tiles.count, INT32_MAX));
            rotary_row_kernel<T><<<grid, 256, 0, stream>>>(
                View(x), View(bias), View(rotary), x.rows, int(width), time, head_dim, dim,
                per_head, offset, tiles, cutlass::FastDivmod(head_dim), cutlass::FastDivmod(time));
        } else {
            int grid = (x.numel() / 2 + 255) / 256;
            rotary_flat_kernel<T><<<grid, 256, 0, stream>>>(View(x), View(bias), View(rotary), x.rows,
                                                       int(width), time, head_dim, dim, per_head,
                                                       offset);
        }
    };
    if (x.type == Type::BF16)
        launch(bf16{});
    else if (x.type == Type::F16)
        launch(__half{});
    else
        launch(float{});
    CUDA(cudaGetLastError());
}
template <class T> __global__ void response_stats(View x, View stats, int time, int channels) {
    using Reduce = cub::BlockReduce<float, 256>;
    __shared__ typename Reduce::TempStorage temp;
    int b = blockIdx.x, c = blockIdx.y;
    float sum = 0;
    for (int t = threadIdx.x; t < time; t += blockDim.x) {
        float v = x.get<T>(b * time + t, c);
        sum += v * v;
    }
    sum = Reduce(temp).Sum(sum);
    if (!threadIdx.x)
        stats.set<float>(b, c, sqrtf(sum));
}
template <class T>
__global__ void response_apply(View x, View y, View gamma, View beta, View stats, int time,
                               int channels) {
    using Reduce = cub::BlockReduce<float, 256>;
    __shared__ typename Reduce::TempStorage temp;
    __shared__ float mean;
    int row = blockIdx.x, b = row / time;
    float sum = 0;
    for (int c = threadIdx.x; c < channels; c += blockDim.x)
        sum += stats.get<float>(b, c);
    sum = Reduce(temp).Sum(sum);
    if (!threadIdx.x)
        mean = sum / channels + 1.e-6f;
    __syncthreads();
    for (int c = threadIdx.x; c < channels; c += blockDim.x) {
        float v = x.get<T>(row, c);
        y.set<T>(row, c,
                 v + beta.get<T>(0, c) + gamma.get<T>(0, c) * v * stats.get<float>(b, c) / mean);
    }
}
void global_response_norm( TensorView x, TensorView y, TensorView gamma,
                          TensorView beta, int batch, int time, TensorView stats) {
    const auto stream = execution_context(x).stream;
    check_output(y);
    check_same(x, y);
    check_same(gamma, y, true);
    check_same(beta, y, true);
    if (batch < 1 || time < 1 || int64_t(batch) * time != y.rows || stats.rows != batch ||
        stats.cols != y.cols || stats.type != Type::F32)
        throw std::invalid_argument("invalid global response normalization workspace");
    dim3 grid(batch, y.cols);
    if (x.type == Type::BF16) {
        response_stats<bf16><<<grid, 256, 0, stream>>>(View(x), View(stats), time, y.cols);
        response_apply<bf16><<<batch * time, 256, 0, stream>>>(View(x), View(y), View(gamma), View(beta),
                                                          View(stats), time, y.cols);
    } else if (x.type == Type::F16) {
        response_stats<__half><<<grid, 256, 0, stream>>>(View(x), View(stats), time, y.cols);
        response_apply<__half><<<batch * time, 256, 0, stream>>>(View(x), View(y), View(gamma),
                                                            View(beta), View(stats), time, y.cols);
    } else {
        response_stats<float><<<grid, 256, 0, stream>>>(View(x), View(stats), time, y.cols);
        response_apply<float><<<batch * time, 256, 0, stream>>>(View(x), View(y), View(gamma),
                                                           View(beta), View(stats), time, y.cols);
    }
    CUDA(cudaGetLastError());
}
template <class T>
__global__ void noise_kernel(View out, int rows, int cols, int time, const int *prefix,
                             const uint64_t *seeds, const uint64_t *subsequences,
                             float temperature) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    int groups = (cols + 3) / 4;
    if (i >= size_t(rows) * groups)
        return;
    int row = i / groups, c = (i % groups) * 4, b = row / time;
    float4 noise{};
    if (row % time >= prefix[b]) {
        curandStatePhilox4_32_10_t state;
        size_t local = size_t(row % time) * groups + c / 4;
        curand_init(seeds[b], local, subsequences[b] * 4, &state);
        noise = curand_normal4(&state);
    }
    const float values[] = {noise.x, noise.y, noise.z, noise.w};
    for (int j = 0; j < 4 && c + j < cols; ++j)
        out.set<T>(row, c + j, float(T(values[j])) * temperature);
}
template <class T>
__global__ void euler_kernel(View x, View velocity, View negative, int rows, int cols, int time,
                             const int *prefix, float delta, float cfg, View out, int offset,
                             int output_time, float low, float high, const void *const *prompt_rows,
                             void *const *history_rows, void *const *value_rows) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(rows) * cols)
        return;
    int row = i / cols, c = i % cols;
    float v = velocity.get<T>(row, c);
    if (negative.data) {
        float difference = float(T(v - negative.get<T>(row, c)));
        v = float(T(v + float(T(difference * cfg))));
    }
    float value = float(T(x.get<T>(row, c) + float(T(v * delta))));
    x.set<T>(row, c, row % time < prefix[row / time] ? 0.f : value);
    if (history_rows && history_rows[row])
        static_cast<T *>(history_rows[row])[c] = T(value);
    int t = row % time - offset;
    if ((out.data || value_rows) && t >= 0 && t < output_time) {
        if (prompt_rows && row % time < prefix[row / time])
            value = float(static_cast<const T *>(prompt_rows[row])[c]);
        float scaled = float(T(float(T(value + 1.f)) * .5f));
        scaled = float(T(scaled * (high - low)));
        float converted = scaled + low;
        out.set<T>(row / time * output_time + t, c, converted);
        if (value_rows && value_rows[row]) static_cast<T *>(value_rows[row])[c] = T(converted);
    }
}
void gaussian_noise( TensorView out, int time, const int *prefix,
                    const uint64_t *seeds, const uint64_t *subsequences, float temperature) {
    const auto stream = execution_context(out).stream;
    check_output(out);
    if (time < 1 || out.rows % time || !prefix || !seeds || !subsequences ||
        !std::isfinite(temperature) || temperature < 0)
        throw std::invalid_argument("invalid noise initialization");
    int grid = (size_t(out.rows) * ((out.cols + 3) / 4) + 255) / 256;
    if (out.type == Type::BF16)
        noise_kernel<bf16><<<grid, 256, 0, stream>>>(View(out), out.rows, out.cols, time, prefix, seeds,
                                                subsequences, temperature);
    else if (out.type == Type::F16)
        noise_kernel<__half><<<grid, 256, 0, stream>>>(View(out), out.rows, out.cols, time, prefix,
                                                  seeds, subsequences, temperature);
    else
        noise_kernel<float><<<grid, 256, 0, stream>>>(View(out), out.rows, out.cols, time, prefix, seeds,
                                                 subsequences, temperature);
    CUDA(cudaGetLastError());
}
void euler_step( TensorView x, TensorView v, TensorView negative, int time,
                const int *prefix, float delta, float cfg, const EulerOutput &out) {
    const auto stream = execution_context(x).stream;
    check_output(x);
    check_same(v, x);
    check_same(negative, x);
    if (!v.data || time < 1 || x.rows % time || !prefix || !(delta > 0) || !std::isfinite(delta) ||
        !std::isfinite(cfg) || cfg < 0 || (cfg > 0 && !negative.data))
        throw std::invalid_argument("invalid Euler integration step");
    if (((out.values.data || out.value_rows) &&
         (out.offset < 0 || out.frames < 1 || out.offset > time - out.frames ||
          !std::isfinite(out.minimum) || !std::isfinite(out.maximum) ||
          !(out.maximum > out.minimum))) ||
        (out.values.data &&
         (out.values.rows != int64_t(x.rows / time) * out.frames || out.values.cols != x.cols ||
          out.values.type != x.type || out.values.data == x.data)))
        throw std::invalid_argument("invalid Euler output range");
    int grid = (x.numel() + 255) / 256;
    if (x.type == Type::BF16)
        euler_kernel<bf16><<<grid, 256, 0, stream>>>(View(x), View(v), View(negative), x.rows, x.cols,
                                                time, prefix, delta, cfg, View(out.values),
                                                out.offset, out.frames, out.minimum, out.maximum,
                                                out.prompt_rows, out.history_rows, out.value_rows);
    else if (x.type == Type::F16)
        euler_kernel<__half><<<grid, 256, 0, stream>>>(View(x), View(v), View(negative), x.rows, x.cols,
                                                  time, prefix, delta, cfg, View(out.values),
                                                  out.offset, out.frames, out.minimum, out.maximum,
                                                  out.prompt_rows, out.history_rows, out.value_rows);
    else
        euler_kernel<float><<<grid, 256, 0, stream>>>(View(x), View(v), View(negative), x.rows, x.cols,
                                                 time, prefix, delta, cfg, View(out.values),
                                                 out.offset, out.frames, out.minimum, out.maximum,
                                                 out.prompt_rows, out.history_rows, out.value_rows);
    CUDA(cudaGetLastError());
}
struct EmbeddingInput {
    View table, residual, residual_default, positions, alpha, segment, norm_weight, norm_bias;
    int time, offset, segment_id, vocabulary, batch_stride;
    float scale, epsilon;
    const int *output_rows, *position_ids, *residual_rows;
    const int *const *id_rows;
    const void *const *residual_sources;
    int position_count, residual_count;
    EmbeddingInput(TensorView t, const EmbeddingFusion &f)
        : table(t), residual(f.residual), residual_default(f.residual_default),
          positions(f.positions), alpha(f.position_alpha),
          segment(f.segment), norm_weight(f.norm_weight), norm_bias(f.norm_bias),
          time(f.sequence_length), offset(f.position_offset), segment_id(f.segment_id),
          vocabulary(t.rows),
          batch_stride(f.batch_stride_rows ? f.batch_stride_rows : f.sequence_length),
          scale(f.table_scale), epsilon(f.epsilon), output_rows(f.output_rows),
          position_ids(f.position_ids), residual_rows(f.residual_rows), id_rows(f.id_rows),
          residual_sources(f.residual_sources),
          position_count(f.positions.rows), residual_count(f.residual.rows) {}
    template <class T> __device__ float get(int id, int row, int col) const {
        if (id < 0 || id >= vocabulary)
            return nanf("");
        float value = float(T(table.get<T>(id, col) * scale));
        const void *source = residual_sources ? residual_sources[row] : nullptr;
        if (source) {
            value = float(T(value + float(static_cast<const T *>(source)[col])));
        } else {
            int r = residual_rows ? residual_rows[row] : row;
            if (r >= 0 && residual.data) {
                if (r >= residual_count) return nanf("");
                value = float(T(value + residual.get<T>(r, col)));
            } else if (residual_default.data) {
                value = float(T(value + residual_default.get<T>(0, col)));
            }
        }
        if (segment.data)
            value = float(T(value + segment.get<T>(segment_id, col)));
        if (positions.data) {
            int p = position_ids ? position_ids[row] : row % time + offset;
            if (p < 0 || p >= position_count)
                return nanf("");
            float position = positions.get<T>(p, col);
            if (alpha.data)
                position = float(T(position * alpha.get<T>(0, 0)));
            value = float(T(value + position));
        }
        return value;
    }
};
template <class T>
__global__ void embedding_kernel(const int *ids, View out, int cols, EmbeddingInput f, int repeat) {
    using Reduce = cub::BlockReduce<float, 256>;
    __shared__ typename Reduce::TempStorage temp;
    __shared__ float mean, inv;
    int row = blockIdx.x, input = row / repeat;
    int id = f.id_rows ? (f.id_rows[input] ? *f.id_rows[input] : 0) : ids[input];
    int output_row =
        f.output_rows ? f.output_rows[row] : (row / f.time) * f.batch_stride + row % f.time;
    if (output_row < 0 || output_row >= out.rows)
        return;
    if (!f.norm_weight.data) {
        for (int c = threadIdx.x; c < cols; c += blockDim.x)
            out.set<T>(output_row, c, f.get<T>(id, row, c));
        return;
    }
    float sum = 0;
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        sum += f.get<T>(id, row, c);
    sum = Reduce(temp).Sum(sum);
    if (!threadIdx.x)
        mean = sum / cols;
    __syncthreads();
    sum = 0;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float d = f.get<T>(id, row, c) - mean;
        sum += d * d;
    }
    sum = Reduce(temp).Sum(sum);
    if (!threadIdx.x)
        inv = rsqrtf(sum / cols + f.epsilon);
    __syncthreads();
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        out.set<T>(output_row, c,
                   (f.get<T>(id, row, c) - mean) * inv * f.norm_weight.get<T>(0, c) +
                       f.norm_bias.get<T>(0, c));
}
void embedding( TensorView table, const int *ids, TensorView out,
               const EmbeddingFusion &f) {
    const auto stream = execution_context(table).stream;
    check_output(out);
    check_same(f.norm_weight, out, true);
    check_same(f.norm_bias, out, true);
    check_same(f.residual_default, out, true);
    int64_t rows = f.input_rows          ? f.input_rows
                   : f.batch_stride_rows ? int64_t(f.batch) * f.sequence_length
                                         : out.rows;
    int64_t required = f.batch_stride_rows
                           ? int64_t(f.batch - 1) * f.batch_stride_rows + f.sequence_length
                           : out.rows;
    if ((!ids && !f.id_rows) || !table.data || table.rows < 1 || table.cols != out.cols ||
        table.type != out.type || f.sequence_length < 1 || rows < 1 || rows > INT32_MAX ||
        required > out.rows || (!f.output_rows && !f.position_ids && rows % f.sequence_length) ||
        f.input_repeat < 1 || rows % f.input_repeat ||
        (!f.output_rows && f.sequence_length % f.input_repeat) || f.input_rows < 0 ||
        (f.input_rows && !f.output_rows && rows > out.rows) || f.position_offset < 0 ||
        f.batch_stride_rows < 0 ||
        (f.batch_stride_rows && (f.batch < 1 || f.batch_stride_rows < f.sequence_length)) ||
        (f.residual.data && (f.residual.type != out.type || f.residual.cols != out.cols ||
                             (!f.residual_rows && f.residual.rows != rows))) ||
        !(f.epsilon > 0) || !std::isfinite(f.epsilon) || !std::isfinite(f.table_scale) ||
        (f.positions.data &&
         (f.positions.cols != out.cols || f.positions.type != out.type ||
          (!f.position_ids && f.sequence_length > f.positions.rows - f.position_offset))) ||
        (f.segment.data && (f.segment_id < 0 || f.segment_id >= f.segment.rows ||
                            f.segment.cols != out.cols || f.segment.type != out.type)) ||
        (f.position_alpha.data &&
         (f.position_alpha.numel() != 1 || f.position_alpha.type != out.type)))
        throw std::invalid_argument("invalid fused embedding");
    if (out.type == Type::BF16)
        embedding_kernel<bf16><<<int(rows), 256, 0, stream>>>(ids, View(out), out.cols,
                                                         EmbeddingInput(table, f), f.input_repeat);
    else if (out.type == Type::F16)
        embedding_kernel<__half><<<int(rows), 256, 0, stream>>>(
            ids, View(out), out.cols, EmbeddingInput(table, f), f.input_repeat);
    else
        embedding_kernel<float><<<int(rows), 256, 0, stream>>>(ids, View(out), out.cols,
                                                          EmbeddingInput(table, f), f.input_repeat);
    CUDA(cudaGetLastError());
}
template <class T>
__global__ void temporal_statistics(View x, View stats, int time, int channels, float eps) {
    using Reduce = cub::BlockReduce<float, 256>;
    __shared__ typename Reduce::TempStorage temp;
    __shared__ float mean;
    int b = blockIdx.x, c = blockIdx.y;
    float sum = 0;
    for (int t = threadIdx.x; t < time; t += blockDim.x)
        sum += x.get<T>(b * time + t, c);
    sum = Reduce(temp).Sum(sum);
    if (!threadIdx.x) {
        mean = sum / time;
        stats.set<float>(b, c, mean);
    }
    __syncthreads();
    sum = 0;
    for (int t = threadIdx.x; t < time; t += blockDim.x) {
        float d = x.get<T>(b * time + t, c) - mean;
        sum += d * d;
    }
    sum = Reduce(temp).Sum(sum);
    if (!threadIdx.x)
        stats.set<float>(b, c + channels, rsqrtf(sum / time + eps));
}
template <class T>
__global__ void temporal_apply(View x, View y, View weight, View bias, View stats, int rows,
                               int channels, int time, Activation activation) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(rows) * channels)
        return;
    int row = i / channels, c = i % channels, b = row / time;
    float value = (x.get<T>(row, c) - stats.get<float>(b, c)) * stats.get<float>(b, c + channels);
    value = float(T(value * weight.get<T>(0, c) + bias.get<T>(0, c)));
    y.set<T>(row, c, activate(value, activation, .1f));
}
void temporal_norm( TensorView x, TensorView y, TensorView weight, TensorView bias,
                   int batch, int time, float eps, Activation activation, TensorView stats) {
    const auto stream = execution_context(x).stream;
    check_output(y);
    check_same(x, y);
    check_same(weight, y, true);
    check_same(bias, y, true);
    if (!weight.data || batch < 1 || time < 1 || int64_t(batch) * time != y.rows || !(eps > 0) ||
        !std::isfinite(eps) || stats.rows != batch || stats.cols != 2 * y.cols ||
        stats.type != Type::F32)
        throw std::invalid_argument("invalid temporal normalization");
    auto launch = [&](auto type) {
        using T = decltype(type);
        temporal_statistics<T>
            <<<dim3(batch, y.cols), 256, 0, stream>>>(View(x), View(stats), time, y.cols, eps);
        temporal_apply<T><<<(y.numel() + 255) / 256, 256, 0, stream>>>(View(x), View(y), View(weight),
                                                                  View(bias), View(stats), y.rows,
                                                                  y.cols, time, activation);
    };
    if (x.type == Type::BF16)
        launch(bf16{});
    else if (x.type == Type::F16)
        launch(__half{});
    else
        launch(float{});
    CUDA(cudaGetLastError());
}
template <class T>
__global__ void gated_kernel(View x, View y, View condition, View residual, int rows, int cols,
                             int time, Activation left, SequenceLayout packed,
                             const void *const *sources, int condition_offset) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(rows) * cols)
        return;
    int row = i / cols, c = i % cols, b = packed.row_batch ? packed.row_batch[row] : row / time;
    const T *source = sources ? static_cast<const T *>(sources[b]) + condition_offset : nullptr;
    float a = float(T(x.get<T>(row, c) + (source ? float(source[c]) : condition.get<T>(b, c))));
    float gate = float(T(x.get<T>(row, cols + c) + (source ? float(source[cols + c]) : condition.get<T>(b, cols + c))));
    a = float(T(activate(a, left, .1f)));
    gate = float(T(activate(gate, Activation::Sigmoid, .1f)));
    y.set<T>(row, c, float(T(a * gate)) + residual.get<T>(row, c));
}
void gated( TensorView x, TensorView y, Activation left, int time,
           TensorView condition, TensorView residual, const SequenceLayout *packed,
           const void *const *condition_sources, int condition_offset) {
    const auto stream = execution_context(x).stream;
    check_output(y);
    check_same(residual, y);
    if (x.type != y.type || x.rows != y.rows || x.cols != int64_t(y.cols) * 2 || time < 1 ||
        (!packed && y.rows % time) || condition_offset < 0 ||
        (packed && packed->rows != y.rows) ||
        (condition.data && (condition.cols != x.cols || condition.type != x.type ||
                            (condition.rows != 1 && condition.rows != y.rows / time))))
        throw std::invalid_argument("invalid conditioned GLU");
    auto launch = [&](auto type) {
        gated_kernel<decltype(type)><<<(y.numel() + 255) / 256, 256, 0, stream>>>(
            View(x), View(y), View(condition), View(residual), y.rows, y.cols, time, left,
            packed ? *packed : SequenceLayout{}, condition_sources, condition_offset);
    };
    if (y.type == Type::BF16)
        launch(bf16{});
    else if (y.type == Type::F16)
        launch(__half{});
    else
        launch(float{});
    CUDA(cudaGetLastError());
}
template <class T> __global__ void mean_kernel(View x, View y, int time, int channels) {
    __shared__ float sums[4][32];
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    int b = blockIdx.y, c = blockIdx.x * 32 + lane;
    float sum = 0;
    if (c < channels)
        for (int t = warp; t < time; t += 4)
            sum += x.get<T>(b * time + t, c) / time;
    sums[warp][lane] = sum;
    __syncthreads();
    if (!warp && c < channels)
        y.set<T>(b, c, sums[0][lane] + sums[1][lane] + sums[2][lane] + sums[3][lane]);
}
void temporal_mean( TensorView x, TensorView y, int time) {
    const auto stream = execution_context(x).stream;
    check_output(y);
    if (time < 1 || x.rows != int64_t(y.rows) * time || x.cols != y.cols || x.type != y.type)
        throw std::invalid_argument("invalid temporal mean");
    dim3 grid((y.cols + 31) / 32, y.rows);
    if (y.type == Type::BF16)
        mean_kernel<bf16><<<grid, 128, 0, stream>>>(View(x), View(y), time, y.cols);
    else if (y.type == Type::F16)
        mean_kernel<__half><<<grid, 128, 0, stream>>>(View(x), View(y), time, y.cols);
    else
        mean_kernel<float><<<grid, 128, 0, stream>>>(View(x), View(y), time, y.cols);
    CUDA(cudaGetLastError());
}
template <class T>
__global__ void residual_split_kernel(View p, View residual, View skip, int rows, int cols,
                                      bool first, bool last, int time, int valid_time, SequenceLayout packed) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(rows) * cols)
        return;
    int row = i / cols, c = i % cols;
    bool valid = packed.row_batch
        ? packed.row_position[row] < packed.lengths[packed.row_batch[row]]
        : !valid_time || row % time < valid_time;
    float value = p.get<T>(row, c + (last ? 0 : cols));
    if (!first)
        value += skip.get<T>(row, c);
    skip.set<T>(row, c, valid ? value : 0.f);
    if (!last)
        residual.set<T>(row, c, valid ? residual.get<T>(row, c) + p.get<T>(row, c) : 0.f);
}
void residual_split( TensorView p, TensorView residual, TensorView skip, bool first,
                    bool last, int time, int valid_time, const SequenceLayout *packed) {
    const auto stream = execution_context(p).stream;
    check_output(skip);
    check_same(residual, skip);
    if (!residual.data || p.rows != skip.rows || p.cols != skip.cols * (last ? 1 : 2) ||
        p.type != skip.type || valid_time < 0 ||
        (packed && packed->rows != skip.rows) ||
        (!packed && valid_time && (time < 1 || skip.rows % time || valid_time > time)))
        throw std::invalid_argument("invalid residual/skip split");
    auto launch = [&](auto type) {
        residual_split_kernel<decltype(type)><<<(skip.numel() + 255) / 256, 256, 0, stream>>>(
            View(p), View(residual), View(skip), skip.rows, skip.cols, first, last, time, valid_time,
            packed ? *packed : SequenceLayout{});
    };
    if (skip.type == Type::BF16)
        launch(bf16{});
    else if (skip.type == Type::F16)
        launch(__half{});
    else
        launch(float{});
    CUDA(cudaGetLastError());
}
} // namespace tts::ops
