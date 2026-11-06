#pragma once
#include "core/quantization.h"
#include "ops/kernels/quant/quant_device.cuh"

namespace tts::quant_device {
template <Type Storage> struct RowOutput {
    static constexpr bool quantized = true;
    static constexpr Type storage_type = Storage;
    unsigned char *values;
    float *scales;
    int rows, group_size, rotation_size;
    explicit RowOutput(RowScaledView output)
        : values(output.values.ptr<unsigned char>()), scales(output.scales.ptr<float>()),
          rows(output.values.rows), group_size(output.group_size), rotation_size(output.rotation_size) {}
    __device__ float scale(int row, float amax) const {
        float value = quant_device::scale<Storage>(amax);
        if (!threadIdx.x) scales[row] = value;
        return 1.f / value;
    }
    __device__ void reset_scales(int row, int columns) const {
        int groups = group_size ? (columns + group_size - 1) / group_size : 1;
        for (int group = threadIdx.x; group < groups; group += blockDim.x)
            scales[size_t(group) * rows + row] = 1.f;
    }
    __device__ void store(int row, int column, int columns, float value, float inverse) const {
        values[size_t(row) * columns + column] = Format<Storage>::encode(value * inverse);
    }
    template <class T, int Count>
    __device__ void store_group(ValuePack<T, Count> value, int row, int column, int columns) const {
        if (rotation_size)
            encode_group<Storage>(rotate_pack(value, column), values, scales, row, column,
                                  rows, columns, group_size);
        else
            encode_group<Storage>(value, values, scales, row, column, rows, columns, group_size);
    }
};
} // namespace tts::quant_device
