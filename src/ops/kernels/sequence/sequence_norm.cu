#include "runtime/cuda_context.h"
#include "ops/kernels/sequence/sequence_internal.cuh"
#include "ops/kernels/sequence/norm_projection.h"
#include "ops/kernels/attention/qkv_writeback.cuh"

namespace tts::ops {
struct NormalizedInput {
    View residual, projection_bias, raw, gate, shift, scale;
    cutlass::FastDivmod sequence;
    const int *source_rows;
    explicit NormalizedInput(const NormFusion &f)
        : residual(f.residual), projection_bias(f.projection_bias), raw(f.raw), gate(f.gate),
          shift(f.shift), scale(f.scale), sequence(f.rows_per_sequence),
          source_rows(f.source_rows) {}
};
template <class T> __device__ const T *norm_row(View v, int row) {
    return v.data ? static_cast<const T *>(v.data) + size_t(v.rows == 1 ? 0 : row) * v.stride
                  : nullptr;
}
template <class T> __device__ T *norm_output(View v, int row) {
    return v.data ? static_cast<T *>(v.data) + size_t(row) * v.stride : nullptr;
}
template <class T> __device__ float norm_read(const T *data, int col) {
    return data ? float(data[col]) : 0.f;
}
// Preserve intermediate storage-type rounding across all cache modes.
template <class T> struct NormRow {
    const T *input, *projection, *residual, *gate, *weight, *bias, *scale, *shift;
    T *output, *raw;
    __device__ NormRow(View x, View y, View w, View b, NormalizedInput f, int row, int out) {
        int sequence = f.sequence.div(row);
        input = norm_row<T>(x, row); projection = norm_row<T>(f.projection_bias, 0);
        residual = norm_row<T>(f.residual, row); gate = norm_row<T>(f.gate, sequence);
        weight = norm_row<T>(w, 0); bias = norm_row<T>(b, 0);
        scale = norm_row<T>(f.scale, sequence); shift = norm_row<T>(f.shift, sequence);
        output = norm_output<T>(y, out); raw = norm_output<T>(f.raw, out);
    }
    __device__ float load(int c) const {
        float value = float(T(float(input[c]) + norm_read(projection, c)));
        if (gate) value = float(T(value * float(gate[c])));
        return float(T(value + norm_read(residual, c)));
    }
    __device__ T transform(int c, float value, float mean, float inverse) const {
        // Preserve the original separate multiply/add rounding even when
        // the quantized consumer makes contraction profitable to the compiler.
        value = __fmul_rn(value - mean, inverse);
        if (weight) value = __fmul_rn(value, float(weight[c]));
        value = float(T(value + norm_read(bias, c)));
        if (scale) value = float(T(value * float(T(1.f + float(scale[c])))));
        return T(value + norm_read(shift, c));
    }
    __device__ void store(int c, float value, float mean, float inverse) const {
        if (raw) raw[c] = T(value);
        output[c] = transform(c, value, mean, inverse);
    }
};
struct DenseRowOutput {
    static constexpr bool quantized = false;
};
constexpr int norm_register_values = 4;
constexpr int projection_register_values = 2;
template <class T, class Output = DenseRowOutput>
__global__ void norm_kernel(View x, View y, View weight, View bias, NormalizedInput f, int cols,
                            float eps, bool register_cached, Output output) {
    __shared__ typename NormReduce::TempStorage temp;
    __shared__ float mean, inv;
    extern __shared__ float values[];
    bool cached = cols <= norm_cache_bytes / int(sizeof(float));
    float registers[norm_register_values] = {};
    int output_row = blockIdx.x, row = f.source_rows ? f.source_rows[output_row] : output_row;
    if (row < 0 || row >= x.rows) {
        for (int c = threadIdx.x; c < cols; c += blockDim.x) {
            y.set<T>(output_row, c, nanf(""));
            if constexpr (Output::quantized)
                output.store(output_row, c, cols, nanf(""), 1.f);
        }
        if constexpr (Output::quantized)
            if (!threadIdx.x) output.scales[output_row] = 1.f;
        return;
    }
    NormRow<T> fusion(x, y, weight, bias, f, row, output_row);
    float sum = 0;
    if (register_cached) {
#pragma unroll
        for (int i = 0; i < norm_register_values; ++i) {
            int c = threadIdx.x + i * norm_threads;
            if (c < cols) registers[i] = fusion.load(c);
            sum += registers[i];
        }
    } else {
        for (int c = threadIdx.x; c < cols; c += blockDim.x) {
            float value = fusion.load(c);
            if (cached) values[c] = value;
            sum += value;
        }
    }
    sum = NormReduce(temp).Sum(sum);
    if (!threadIdx.x)
        mean = sum / cols;
    __syncthreads();
    sum = 0;
    float row_mean = mean;
    if (register_cached) {
#pragma unroll
        for (int i = 0; i < norm_register_values; ++i)
            if (threadIdx.x + i * norm_threads < cols) {
                float delta = registers[i] - row_mean;
                sum += delta * delta;
            }
    } else {
        for (int c = threadIdx.x; c < cols; c += blockDim.x) {
            float value = cached ? values[c] : fusion.load(c);
            float delta = value - row_mean;
            sum += delta * delta;
        }
    }
    sum = NormReduce(temp).Sum(sum);
    if (!threadIdx.x)
        inv = rsqrtf(sum / cols + eps);
    __syncthreads();
    float row_inv = inv;
    if constexpr (Output::quantized) {
        float amax = 0;
        if (register_cached) {
#pragma unroll
            for (int i = 0; i < norm_register_values; ++i) {
                int c = threadIdx.x + i * norm_threads;
                if (c < cols) {
                    float value = registers[i];
                    T normalized = fusion.transform(c, value, row_mean, row_inv);
                    if (fusion.raw) fusion.raw[c] = T(value);
                    if (fusion.output) fusion.output[c] = normalized;
                    registers[i] = float(normalized);
                    amax = fmaxf(amax, fabsf(registers[i]));
                }
            }
        } else {
            for (int c = threadIdx.x; c < cols; c += norm_threads) {
                float value = cached ? values[c] : fusion.load(c);
                T normalized = fusion.transform(c, value, row_mean, row_inv);
                float result = float(normalized);
                if (cached) {
                    if (fusion.raw) fusion.raw[c] = T(value);
                    if (fusion.output) fusion.output[c] = normalized;
                    values[c] = result;
                }
                amax = fmaxf(amax, fabsf(result));
            }
        }
        amax = quant_device::maximum<norm_threads>(amax);
        float reciprocal = output.scale(output_row, amax);
        if (register_cached) {
#pragma unroll
            for (int i = 0; i < norm_register_values; ++i) {
                int c = threadIdx.x + i * norm_threads;
                if (c < cols) output.store(output_row, c, cols, registers[i], reciprocal);
            }
        } else {
            for (int c = threadIdx.x; c < cols; c += norm_threads) {
                float result;
                if (cached) result = values[c];
                else {
                    float value = fusion.load(c);
                    T normalized = fusion.transform(c, value, row_mean, row_inv);
                    if (fusion.raw) fusion.raw[c] = T(value);
                    if (fusion.output) fusion.output[c] = normalized;
                    result = float(normalized);
                }
                output.store(output_row, c, cols, result, reciprocal);
            }
        }
    } else {
        if (register_cached) {
    #pragma unroll
            for (int i = 0; i < norm_register_values; ++i) {
                int c = threadIdx.x + i * norm_threads;
                if (c < cols) fusion.store(c, registers[i], row_mean, row_inv);
            }
        } else {
            for (int c = threadIdx.x; c < cols; c += blockDim.x) {
                float value = cached ? values[c] : fusion.load(c);
                fusion.store(c, value, row_mean, row_inv);
            }
        }
    }
}

template <class T> struct DenseProjection {
    using Acc = float;
    static constexpr bool quantized = false;
    static constexpr int channels_per_block = norm_threads / 32;
    const T *weight;
    template <class U>
    __device__ float dot(const U *activation, int channel, int lane, int cols) const {
        return warp_dot_four(activation, weight + size_t(channel) * cols, cols, lane, false);
    }
};
template <Type Storage> struct QuantProjection {
    using Format = quant_device::Format<Storage>;
    using Acc = typename Format::Acc;
    static constexpr bool quantized = true;
    static constexpr Type storage = Storage;
    static constexpr int channels_per_block = norm_threads / 32;
    const unsigned char *weight;
    const float *weight_scales;
    unsigned char *encoded;
    float *encoded_scales;
    explicit QuantProjection(RowScaledView w, RowScaledView q)
        : weight(w.values.ptr<unsigned char>()), weight_scales(w.scales.ptr<float>()),
          encoded(q.values.ptr<unsigned char>()), encoded_scales(q.scales.ptr<float>()) {}
    template <class T>
    __device__ Acc dot(const T *activation, int channel, int lane, int cols) const {
        Acc sum = 0;
#pragma unroll 4
        for (int c = lane * 4; c < cols; c += 128) {
            unsigned b = *reinterpret_cast<const unsigned *>(weight + size_t(channel) * cols + c);
            if constexpr (Storage == Type::I8) {
                unsigned a = *reinterpret_cast<const unsigned *>(
                    reinterpret_cast<const unsigned char *>(activation) + c);
                sum = Format::dot(a, b, sum);
            } else {
                // Decode the input once while staging the CTA's normalized row.
                // Each output warp only decodes its weight operand.
                __nv_fp8x4_e4m3 packed;
                packed.__x = b;
                float4 w = float4(packed), a = load_four(activation + c);
                sum = fmaf(a.x, w.x, sum);
                sum = fmaf(a.y, w.y, sum);
                sum = fmaf(a.z, w.z, sum);
                sum = fmaf(a.w, w.w, sum);
            }
        }
        return sum;
    }
    __device__ float result(Acc value, int channel, float scale) const {
        return value * weight_scales[channel] * scale;
    }
};
// Each CTA keeps a normalized row in shared memory and projects eight
// output channels. Only the first CTA stores the persistent residual result;
// all CTAs read the original operands, so those buffers must be disjoint.
template <class T, class Projection = DenseProjection<T>>
__global__ void norm_project_kernel(View input, View normalized, View norm_weight, View norm_bias,
                                    NormalizedInput f, Projection projection, View output, Finish epilogue,
                                    int channels, int cols, float epsilon,
                                    QkvProjectionWriteback writeback) {
    __shared__ typename NormReduce::TempStorage temporary;
    __shared__ float mean, inverse;
    __shared__ __align__(16) T activation[norm_threads * projection_register_values];
    __shared__ float token_scale;
    int row = blockIdx.y, lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    int channel = blockIdx.x * Projection::channels_per_block + warp;
    NormRow<T> fusion(input, normalized, norm_weight, norm_bias, f, row, row);
    float values[projection_register_values], sum = 0;
#pragma unroll
    for (int i = 0; i < projection_register_values; ++i) {
        int col = threadIdx.x + i * norm_threads;
        values[i] = col < cols ? fusion.load(col) : 0.f;
        sum += values[i];
    }
    sum = NormReduce(temporary).Sum(sum);
    if (!threadIdx.x) mean = sum / cols;
    __syncthreads();
    float mu = mean;
    sum = 0;
#pragma unroll
    for (int i = 0; i < projection_register_values; ++i)
        if (threadIdx.x + i * norm_threads < cols) {
            float delta = values[i] - mu;
            sum += delta * delta;
        }
    sum = NormReduce(temporary).Sum(sum);
    if (!threadIdx.x) inverse = rsqrtf(sum / cols + epsilon);
    __syncthreads();
    float inv = inverse, amax = 0;
    T rounded[projection_register_values];
#pragma unroll
    for (int i = 0; i < projection_register_values; ++i) {
        int col = threadIdx.x + i * norm_threads;
        if (col < cols) {
            T value = fusion.transform(col, values[i], mu, inv);
            if constexpr (Projection::quantized) {
                rounded[i] = value;
                amax = fmaxf(amax, fabsf(float(value)));
            } else {
                activation[col] = value;
            }
            if (!blockIdx.x) {
                if (fusion.raw) fusion.raw[col] = T(values[i]);
                if (fusion.output) fusion.output[col] = value;
            }
        }
    }
    if constexpr (Projection::quantized) {
        amax = quant_device::maximum<norm_threads>(amax);
        float scale = quant_device::scale<Projection::storage>(amax), reciprocal = 1.f / scale;
        if (!threadIdx.x) {
            token_scale = scale;
            if (!blockIdx.x && projection.encoded_scales) projection.encoded_scales[row] = scale;
        }
        if constexpr (Projection::storage == Type::FP8) {
#pragma unroll
            for (int i = 0; i < projection_register_values; i += 2) {
                int c = threadIdx.x + i * norm_threads;
                float a = c < cols ? float(rounded[i]) * reciprocal : 0.f;
                float b = i + 1 < projection_register_values && c + norm_threads < cols
                    ? float(rounded[i + 1]) * reciprocal : 0.f;
                unsigned short q = Projection::Format::pair(make_float2(a, b));
                __half2 decoded = static_cast<__half2>(__nv_cvt_fp8x2_to_halfraw2(q, __NV_E4M3));
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    int col = c + j * norm_threads;
                    if (i + j < projection_register_values && col < cols) {
                        activation[col] = T(j ? __high2half(decoded) : __low2half(decoded));
                        if (!blockIdx.x && projection.encoded)
                            projection.encoded[size_t(row) * cols + col] = q >> (j * 8);
                    }
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < projection_register_values; ++i) {
                int col = threadIdx.x + i * norm_threads;
                if (col < cols) {
                    unsigned char q = Projection::Format::encode(float(rounded[i]) * reciprocal);
                    reinterpret_cast<unsigned char *>(activation)[col] = q;
                    if (!blockIdx.x && projection.encoded)
                        projection.encoded[size_t(row) * cols + col] = q;
                }
            }
        }
    }
    __syncthreads();
    typename Projection::Acc accumulator = channel < channels
        ? projection.dot(activation, channel, lane, cols) : 0;
    if constexpr (std::is_same_v<typename Projection::Acc, int32_t>) {
        accumulator = __reduce_add_sync(0xffffffffu, accumulator);
    } else {
        for (int delta = 16; delta; delta /= 2)
            accumulator += __shfl_down_sync(0xffffffffu, accumulator, delta);
    }
    if (channel < channels && !lane) {
        float result;
        if constexpr (Projection::quantized) result = projection.result(accumulator, channel, token_scale);
        else result = accumulator;
        if (writeback.query)
            writeback.store(T(result), row, channel);
        else
            output.set<T>(row, channel, epilogue.apply<T>(float(T(result)), row, channel));
    }

}
static void check_norm_shape(TensorView x, TensorShape y, Type type, void *retained,
                             TensorView w, TensorView bias,
                float eps, const NormFusion &f) {
    check_same(w, y, type, true);
    check_same(bias, y, type, true);
    if (f.source_rows) {
        if (x.type != type || x.cols != y.cols || x.data == retained || f.gate.data ||
            f.shift.data || f.scale.data)
            throw std::invalid_argument("invalid indexed normalization");
    } else
        check_same(x, y, type);
    check_same(f.residual, x);
    check_same(f.projection_bias, y, type, true);
    check_same(f.raw, y, type);
    if (f.rows_per_sequence < 1 || y.rows % f.rows_per_sequence)
        throw std::invalid_argument("invalid normalization sequence length");
    for (auto v : {f.gate, f.shift, f.scale})
        if (v.data && (v.type != type || v.cols != y.cols ||
                       (v.rows != 1 && v.rows != y.rows / f.rows_per_sequence)))
            throw std::invalid_argument("invalid normalization modulation");
    if (!x.data || !(eps > 0) || !std::isfinite(eps))
        throw std::invalid_argument("invalid layer normalization");
}
static void check_norm(TensorView x, TensorView y, TensorView w, TensorView bias,
                       float eps, const NormFusion &f) {
    check_output(y);
    check_norm_shape(x, TensorShape(y), y.type, y.data, w, bias, eps, f);
}
void layer_norm( TensorView x, TensorView y, TensorView w, TensorView bias,
                float eps, const NormFusion &f) {
    const auto stream = execution_context(x).stream;
    check_norm(x, y, w, bias, eps, f);
    bool register_cached = y.cols <= norm_threads * norm_register_values;
    size_t shared_bytes = !register_cached && y.cols <= norm_cache_bytes / int(sizeof(float))
                              ? size_t(y.cols) * sizeof(float) : 0;
    if (y.type == Type::BF16)
        norm_kernel<bf16><<<y.rows, norm_threads, shared_bytes, stream>>>(
            View(x), View(y), View(w), View(bias), NormalizedInput(f), y.cols, eps, register_cached, DenseRowOutput{});
    else if (y.type == Type::F16)
        norm_kernel<__half><<<y.rows, norm_threads, shared_bytes, stream>>>(
            View(x), View(y), View(w), View(bias), NormalizedInput(f), y.cols, eps, register_cached, DenseRowOutput{});
    else
        norm_kernel<float><<<y.rows, norm_threads, shared_bytes, stream>>>(
            View(x), View(y), View(w), View(bias), NormalizedInput(f), y.cols, eps, register_cached, DenseRowOutput{});
    CUDA(cudaGetLastError());
}
void layer_norm_quantize( TensorView x, RowScaledView q, TensorView w,
                         TensorView bias, float eps, const NormFusion &f, TensorView retained) {
    const auto stream = execution_context(x).stream;
    auto shape = check_quantized_output(x, q, retained);
    check_norm_shape(x, shape, x.type, retained.data, w, bias, eps, f);
    check_quantized_aliases(q, {x, retained, w, bias, f.residual, f.projection_bias, f.raw,
                               f.gate, f.shift, f.scale});
    check_retained_raw(retained, f.raw, {w, bias, f.projection_bias, f.gate, f.shift, f.scale});
    for (auto output : {retained, f.raw}) {
        check_row_alias(output, x, !f.source_rows);
        check_row_alias(output, f.residual, !f.source_rows);
    }
    bool register_cached = shape.cols <= norm_threads * norm_register_values;
    size_t shared = !register_cached && shape.cols <= norm_cache_bytes / int(sizeof(float))
                        ? size_t(shape.cols) * sizeof(float) : 0;
    auto launch = [&](auto type, auto output) {
        norm_kernel<decltype(type), decltype(output)><<<shape.rows, norm_threads, shared, stream>>>(
            View(x), View(retained), View(w), View(bias), NormalizedInput(f), shape.cols, eps,
            register_cached, output);
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
int normalized_linear_capacity(int multiprocessors, int latency_warps,
                               int &writeback_capacity) {
    int half_blocks = 0, bf16_blocks = 0;
    CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&half_blocks, norm_project_kernel<__half>,
                                                      norm_threads, 0));
    CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bf16_blocks, norm_project_kernel<bf16>,
                                                      norm_threads, 0));
    int resident = std::min(half_blocks, bf16_blocks);
    constexpr int warps = norm_threads / 32;
    // Enough independent warps hide the dependent SIMT FMA chain. More
    // CTAs repeat both norm reductions and operand reads before projection.
    // This is a conservative latency filter, not a speed guarantee.
    int latency_blocks = (std::max(0, latency_warps) + warps - 1) / warps;
    writeback_capacity = multiprocessors * std::min(resident, latency_blocks);
    return multiprocessors * resident;
}
bool normalized_linear(cudaStream_t stream, TensorView x, TensorView normalized,
                       TensorView norm_weight, TensorView norm_bias, float epsilon,
                       const NormFusion &f, TensorView weight, TensorView output,
                       const Epilogue &e, int resident_blocks,
                       const QkvProjectionOutput *writeback) {
    if (e.packed || e.condition_sources || e.output_rows) return false;
    if (writeback) writeback->validate(x, weight);
    constexpr int warps = norm_threads / 32;
    size_t blocks = size_t((weight.rows + warps - 1) / warps) * x.rows;
    if (!x.data || !normalized.data || !weight.data || !output.data ||
        x.rows < 1 || x.cols < 1 || weight.rows < 1 || x.cols > norm_threads * projection_register_values ||
        x.rows > 65535 || blocks > size_t(std::max(0, resident_blocks)) ||
        (x.type != Type::F16 && x.type != Type::BF16) || weight.type != x.type ||
        normalized.type != x.type || output.type != x.type || !weight.dense() ||
        weight.cols != x.cols || output.rows != x.rows ||
        (!writeback && output.cols != weight.rows) ||
        (writeback && (output.data != writeback->query.data ||
                       output.cols != writeback->query.cols ||
                       output.ld() != writeback->query.ld() || e.bias.data ||
                       e.activation != Activation::Identity)) ||
        normalized.rows != x.rows || normalized.cols != x.cols || f.source_rows ||
        (e.activation != Activation::Identity && e.activation != Activation::Relu) ||
        e.residual.data || e.gate.data || e.condition.data || e.raw.data || e.residual_rows ||
        e.input_repeat != 1 || e.value_scale != 1 || e.output_scale != 1)
        return false;
    for (auto source : {x, f.residual, f.projection_bias, f.gate, f.scale, f.shift,
                        norm_weight, norm_bias, weight, e.bias,
                        writeback ? writeback->bias : TensorView{}})
        if (normalized.overlaps(source) || output.overlaps(source) || f.raw.overlaps(source))
            return false;
    if (normalized.overlaps(output) || f.raw.overlaps(output)) return false;
    check_norm(x, normalized, norm_weight, norm_bias, epsilon, f);
    check_epilogue(e, output);
    dim3 grid((weight.rows + warps - 1) / warps, x.rows);
    QkvProjectionWriteback sink;
    if (writeback) sink = QkvProjectionWriteback(*writeback);
    if (x.type == Type::F16)
        norm_project_kernel<__half><<<grid, norm_threads, 0, stream>>>(
            View(x), View(normalized), View(norm_weight), View(norm_bias), NormalizedInput(f),
            DenseProjection<__half>{weight.ptr<__half>()}, View(output), Finish(e), weight.rows, x.cols, epsilon, sink);
    else
        norm_project_kernel<bf16><<<grid, norm_threads, 0, stream>>>(
            View(x), View(normalized), View(norm_weight), View(norm_bias), NormalizedInput(f),
            DenseProjection<bf16>{weight.ptr<bf16>()}, View(output), Finish(e), weight.rows, x.cols, epsilon, sink);
    CUDA(cudaGetLastError());
    return true;
}
template <Type Format>
static int quantized_projection_residency() {
    int half = 0, bfloat = 0;
    CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &half, norm_project_kernel<__half, QuantProjection<Format>>, norm_threads, 0));
    CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &bfloat, norm_project_kernel<bf16, QuantProjection<Format>>, norm_threads, 0));
    return std::min(half, bfloat);
}
int normalized_quantized_linear_capacity(Type format, int multiprocessors) {
    int resident = format == Type::I8 ? quantized_projection_residency<Type::I8>()
                                      : quantized_projection_residency<Type::FP8>();
    return multiprocessors * resident;
}
bool normalized_quantized_linear(cudaStream_t stream, TensorView x, TensorView normalized,
                                TensorView norm_weight, TensorView norm_bias, float epsilon,
                                const NormFusion &f, RowScaledView weight, TensorView output,
                                const Epilogue &e, int resident_blocks, RowScaledView encoded,
                                const QkvProjectionOutput *writeback) {
    if (e.packed || e.condition_sources || e.output_rows) return false;
    weight.validate();
    if (writeback) writeback->validate(x, weight.values);
    int n = weight.values.rows, k = weight.values.cols;
    if (!x.data || !weight.values.data || !weight.scales.data || !output.data ||
        x.rows < 1 || x.rows > 65535 || n < 1 || k < 4 || k % 4 ||
        reinterpret_cast<uintptr_t>(weight.values.data) % alignof(unsigned) ||
        k > norm_threads * projection_register_values || x.cols != k ||
        (x.type != Type::F16 && x.type != Type::BF16) || output.type != x.type ||
        output.rows != x.rows || (!writeback && output.cols != n) || f.source_rows ||
        (writeback && (output.data != writeback->query.data ||
                       output.cols != writeback->query.cols ||
                       output.ld() != writeback->query.ld() || e.bias.data ||
                       e.activation != Activation::Identity)) ||
        size_t(x.rows) * ((n + norm_threads / 32 - 1) / (norm_threads / 32)) > size_t(std::max(0, resident_blocks)) ||
        (normalized.data && (normalized.rows != x.rows || normalized.cols != k || normalized.type != x.type)) ||
        (e.activation != Activation::Identity && e.activation != Activation::Relu) ||
        e.residual.data || e.gate.data || e.condition.data || e.raw.data || e.residual_rows ||
        e.input_repeat != 1 || e.value_scale != 1 || e.output_scale != 1)
        return false;
    for (auto source : {x, f.residual, f.projection_bias, f.gate, f.scale, f.shift,
                        norm_weight, norm_bias, weight.values, weight.scales, e.bias,
                        writeback ? writeback->bias : TensorView{}})
        if (normalized.overlaps(source) || output.overlaps(source) || f.raw.overlaps(source))
            return false;
    if (normalized.overlaps(output) || f.raw.overlaps(output) || f.raw.overlaps(normalized)) return false;
    check_norm_shape(x, TensorShape(x), x.type, normalized.data, norm_weight, norm_bias, epsilon, f);
    check_epilogue(e, output);
    if (encoded.values.data || encoded.scales.data) {
        check_quantized_output(x, encoded, normalized);
        if (encoded.values.type != weight.values.type || encoded.values.rows != x.rows || encoded.values.cols != k)
            throw std::invalid_argument("normalized projection quantized output mismatch");
        check_quantized_aliases(encoded, {x, normalized, norm_weight, norm_bias, f.residual,
            f.projection_bias, f.raw, f.gate, f.shift, f.scale, weight.values, weight.scales, output, e.bias,
            writeback ? writeback->bias : TensorView{}});
    }
    QkvProjectionWriteback sink;
    if (writeback) sink = QkvProjectionWriteback(*writeback);
    auto launch = [&](auto type, auto projection) {
        norm_project_kernel<decltype(type), decltype(projection)>
            <<<dim3((n + decltype(projection)::channels_per_block - 1) / decltype(projection)::channels_per_block, x.rows),
                norm_threads, 0, stream>>>(View(x), View(normalized), View(norm_weight), View(norm_bias),
                    NormalizedInput(f), projection, View(output), Finish(e), n, k, epsilon, sink);
    };
    auto typed = [&](auto type) {
        if (weight.values.type == Type::I8) launch(type, QuantProjection<Type::I8>(weight, encoded));
        else launch(type, QuantProjection<Type::FP8>(weight, encoded));
    };
    if (x.type == Type::F16) typed(__half{}); else typed(bf16{});
    CUDA(cudaGetLastError());
    return true;
}
} // namespace tts::ops
