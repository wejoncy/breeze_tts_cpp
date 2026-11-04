#pragma once
#include "ops/sequence.h"
#include "runtime/device.h"
#include "ops/kernels/common/cuda_math.cuh"
#include "ops/kernels/common/sequence_address.cuh"
#include "ops/kernels/common/activation.cuh"
#include "ops/kernels/quant/quant_device.cuh"
#include <cub/block/block_reduce.cuh>
#include <cutlass/fast_math.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <type_traits>

namespace tts::ops {
using bf16 = __nv_bfloat16;
struct View {
    void *data;
    int stride, rows;
    __host__ explicit View(TensorView v) : data(v.data), stride(v.ld()), rows(v.rows) {}
    template <class T> __device__ float get(int row, int col) const {
        return data
                   ? float(static_cast<const T *>(data)[size_t(rows == 1 ? 0 : row) * stride + col])
                   : 0.f;
    }
    template <class T> __device__ void set(int row, int col, float v) const {
        if (data)
            static_cast<T *>(data)[size_t(row) * stride + col] = T(v);
    }
};
struct Finish {
    View bias, residual, raw, gate, condition;
    int rows_per_sequence, valid_rows_per_sequence;
    float value_scale, output_scale, scale, alpha;
    Activation activation;
    const int *residual_rows;
    struct Row {
        int batch, residual;
    };
    explicit Finish(const Epilogue &e)
        : bias(e.bias), residual(e.residual), raw(e.raw), gate(e.gate), condition(e.condition),
          rows_per_sequence(e.rows_per_sequence), valid_rows_per_sequence(e.valid_rows_per_sequence),
          value_scale(e.value_scale),
          output_scale(e.output_scale), scale(e.residual_scale), alpha(e.alpha),
          activation(e.activation), residual_rows(e.residual_rows) {}
    __device__ Row prepare_row(int row, int batch) const {
        return {batch, residual_rows ? residual_rows[row] : row};
    }
    template <class T> __device__ void store(View output, int row, int col, float value) const {
        output.set<T>(row, col, value);
    }
    template <class T>
    __device__ __forceinline__ float apply(float value, int row, int col,
                                           const Row *prepared = nullptr) const {
        if (valid_rows_per_sequence && row % rows_per_sequence >= valid_rows_per_sequence) {
            raw.set<T>(row, col, 0.f);
            return 0.f;
        }
        value = float(T(value + bias.get<T>(0, col)));
        if (gate.data)
            value = float(
                T(value * gate.get<T>(prepared ? prepared->batch : row / rows_per_sequence, col)));
        int r = prepared ? prepared->residual : residual_rows ? residual_rows[row] : row;
        float skip = 0;
        if (residual.data && r >= 0)
            skip = r < residual.rows ? residual.get<T>(r, col) : nanf("");
        value = float(T(value_scale * value + scale * skip));
        if (condition.data)
            value =
                float(T(value + condition.get<T>(
                                    prepared ? prepared->batch : row / rows_per_sequence, col)));
        raw.set<T>(row, col, value);
        return activate(float(T(value * output_scale)), activation, alpha);
    }
};
// Only indirect pointwise consumers carry sequence/source/sink addressing.
// The dense Finish above retains its original parameter footprint and arithmetic.
struct PackedFinish : Finish {
    SequenceLayout packed{};
    const void *const *condition_sources;
    void *const *output_rows;
    struct Row { Finish::Row base; const void *condition; bool valid; };
    explicit PackedFinish(const Epilogue &e)
        : Finish(e), packed(e.packed ? *e.packed : SequenceLayout{}),
          condition_sources(e.condition_sources), output_rows(e.output_rows) {}
    __device__ Row prepare_row(int row, int batch) const {
        if (packed.row_batch) batch = packed.row_batch[row];
        bool valid = !packed.row_batch || packed.row_position[row] < packed.lengths[batch];
        return {{batch, residual_rows ? residual_rows[row] : row},
                condition_sources ? condition_sources[batch] : nullptr, valid};
    }
    template <class T> __device__ float apply(float value, int row, int col,
                                             const Row *prepared = nullptr) const {
        Row selected = prepared ? *prepared : prepare_row(row, row / rows_per_sequence);
        if (!selected.valid) { raw.set<T>(row, col, 0.f); return 0.f; }
        Finish local = *this;
        if (condition_sources) {
            local.condition.data = const_cast<void *>(selected.condition);
            local.condition.rows = 1;
        }
        return local.apply<T>(value, row, col, &selected.base);
    }
    template <class T> __device__ void store(View output, int row, int col, float value) const {
        if (output_rows) static_cast<T *>(output_rows[row])[col] = T(value);
        else output.set<T>(row, col, value);
    }
};
// Four independent columns per thread, with power-of-two thread groups
// spanning rows. This mapping also covers narrow/odd rows and tiled wide rows.
struct RowTiles {
    int row_shift = 0, rows_per_tile, columns_per_tile;
    uint64_t count;
    cutlass::FastDivmodU64 column_tiles;
    explicit RowTiles(int rows, int cols) {
        while (row_shift < 8 && (4 << row_shift) < cols)
            ++row_shift;
        rows_per_tile = 256 >> row_shift;
        columns_per_tile = 4 << row_shift;
        uint64_t n = (uint64_t(cols) + columns_per_tile - 1) / columns_per_tile;
        column_tiles = cutlass::FastDivmodU64(n);
        count = ((uint64_t(rows) + rows_per_tile - 1) / rows_per_tile) * n;
    }
    // A row spans at least four warps and fills all four value slots;
    // packing no more than two rows per CTA conserves grid parallelism.
    bool full_rows(int rows, int cols) const {
        return rows >= rows_per_tile && rows_per_tile <= 256 / (4 * 32) &&
               cols % columns_per_tile == 0;
    }
    __device__ void coordinates(uint64_t tile, int &row, int &col) const {
        uint64_t column;
        row = int(column_tiles.divmod(column, tile)) * rows_per_tile;
        col = int(column) * columns_per_tile;
    }
};
template <Type Storage> struct QuantizedRowOutput {
    static constexpr bool quantized = true;
    static constexpr Type storage_type = Storage;
    unsigned char *values;
    float *scales;
    explicit QuantizedRowOutput(RowScaledView q)
        : values(q.values.ptr<unsigned char>()), scales(q.scales.ptr<float>()) {}
    __device__ float scale(int row, float amax) const {
        float value = quant_device::scale<Storage>(amax);
        if (!threadIdx.x) scales[row] = value;
        return 1.f / value;
    }
    __device__ void store(int row, int col, int cols, float value, float inverse) const {
        values[size_t(row) * cols + col] = quant_device::Format<Storage>::encode(value * inverse);
    }
};
constexpr int norm_threads = 256;
using NormReduce = cub::BlockReduce<float, norm_threads>;
// Stay below CUDA's portable default 48 KiB block shared-memory budget,
// including CUB's static storage and scalar/alignment overhead.
constexpr int norm_cache_bytes = 48 * 1024 - sizeof(NormReduce::TempStorage) - 128;
static void check_same(TensorView v, TensorShape target, Type type, bool vector = false) {
    if (v.data &&
        (v.type != type || (vector ? v.numel() != size_t(target.cols)
                                    : v.rows != target.rows || v.cols != target.cols)))
        throw std::invalid_argument("sequence operation shape/type mismatch");
}
static void check_same(TensorView v, TensorView target, bool vector = false) {
    check_same(v, TensorShape(target), target.type, vector);
}
static void check_output(TensorView out) {
    if (!out.data || !out.rows || !out.cols ||
        (out.type != Type::BF16 && out.type != Type::F16 && out.type != Type::F32))
        throw std::invalid_argument("sequence operation requires nonempty BF16/F32 output");
}
static void check_epilogue(const Epilogue &e, TensorShape out, Type type) {
    if (e.packed && (e.packed->rows != out.rows || e.packed->sequences < 1 ||
        e.packed->max_length < 1 || e.packed->max_length > out.rows ||
        !e.packed->row_batch || !e.packed->row_position || !e.packed->cu_seqlens || !e.packed->lengths))
        throw std::invalid_argument("invalid packed epilogue metadata");
    check_same(e.bias, out, type, true);
    if (e.residual_rows) {
        if (!e.residual.data || e.residual.type != type || e.residual.cols != out.cols)
            throw std::invalid_argument("invalid indexed residual");
    } else
        check_same(e.residual, out, type);
    check_same(e.raw, out, type);
    if (e.rows_per_sequence < 1 || (!e.packed && out.rows % e.rows_per_sequence) ||
        (e.packed && e.valid_rows_per_sequence) ||
        e.valid_rows_per_sequence < 0 || e.valid_rows_per_sequence > e.rows_per_sequence)
        throw std::invalid_argument("invalid epilogue sequence length");
    int sequences = e.packed ? e.packed->sequences : out.rows / e.rows_per_sequence;
    if (e.gate.data && (e.gate.type != type || e.gate.cols != out.cols ||
                        (e.gate.rows != 1 && e.gate.rows != sequences)))
        throw std::invalid_argument("invalid epilogue gate");
    if (e.condition.data &&
        (e.condition.type != type || e.condition.cols != out.cols ||
         (e.condition.rows != 1 && e.condition.rows != sequences)))
        throw std::invalid_argument("invalid epilogue condition");
}
static void check_epilogue(const Epilogue &e, TensorView out) {
    check_epilogue(e, TensorShape(out), out.type);
}
static TensorShape check_quantized_output(TensorView x, RowScaledView q, TensorView retained) {
    q.validate();
    if (!q.values.data || !q.scales.data || q.values.rows < 1 || q.values.cols < 1 ||
        (x.type != Type::F16 && x.type != Type::BF16))
        throw std::invalid_argument("quantized sequence output requires F16/BF16 input");
    if (retained.data) {
        check_output(retained);
        check_same(retained, TensorShape(q.values), x.type);
    }
    if (q.values.overlaps(q.scales))
        throw std::invalid_argument("quantized values/scales overlap");
    return TensorShape(q.values);
}
static void check_quantized_aliases(RowScaledView q, std::initializer_list<TensorView> sources) {
    for (auto source : sources)
        if (q.values.overlaps(source) || q.scales.overlaps(source))
            throw std::invalid_argument("quantized output overlaps sequence storage");
}
// A row kernel can overwrite an input row after consuming it. Broadcast,
// permuted/indexed, repeated, and partially overlapping rows cannot be updated
// concurrently without a separate buffer supplied by the caller.
static void check_row_alias(TensorView output, TensorView source, bool same_rows) {
    if (output.overlaps(source) &&
        (!same_rows || output.data != source.data || output.rows != source.rows ||
         output.cols != source.cols || output.ld() != source.ld()))
        throw std::invalid_argument("quantized sequence output has unsafe row alias");
}
static void check_retained_raw(TensorView retained, TensorView raw,
                               std::initializer_list<TensorView> broadcast) {
    if (retained.overlaps(raw))
        throw std::invalid_argument("raw and retained sequence outputs overlap");
    for (auto source : broadcast) {
        check_row_alias(retained, source, false);
        check_row_alias(raw, source, false);
    }
}

} // namespace tts::ops
