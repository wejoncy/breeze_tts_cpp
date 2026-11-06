#pragma once
#include "core/buffer.h"
#include "core/quantization.h"
#include <algorithm>

namespace tts {
// Reusable activation storage. Layers describe their requirements during
// planning; the same arena storage serves every projection during execution.
class QuantizationWorkspace {
  public:
    struct Capacity {
        int columns = 0, scales_per_row = 0;
        void include(Capacity other) {
            columns = std::max(columns, other.columns);
            scales_per_row = std::max(scales_per_row, other.scales_per_row);
        }
        static Capacity matrix(int columns, int group_size) {
            return {columns, group_size ? (columns + group_size - 1) / group_size : 1};
        }
    };

  private:
    Buffer values_, scales_;
    int rows_ = 0;
    Capacity capacity_;

  public:
    QuantizationWorkspace() = default;
    QuantizationWorkspace(int rows, Capacity capacity) : rows_(rows), capacity_(capacity) {
        if (rows < 1 || capacity.columns < 0 || capacity.scales_per_row < 0 ||
            bool(capacity.columns) != bool(capacity.scales_per_row))
            throw std::invalid_argument("invalid quantization workspace capacity");
        if (size_t(rows) > SIZE_MAX / size_t(std::max(1, capacity.columns)) ||
            size_t(rows) > SIZE_MAX / sizeof(float) / size_t(std::max(1, capacity.scales_per_row)))
            throw std::overflow_error("quantization workspace size");
        values_ = Buffer(size_t(rows) * capacity.columns);
        scales_ = Buffer(size_t(rows) * capacity.scales_per_row * sizeof(float));
    }
    RowScaledView view(int rows, int columns, Type format, QuantizationGroups layout) const {
        auto required = Capacity::matrix(columns, layout.activation);
        if (rows < 1 || rows > rows_ || columns < 1 || columns > capacity_.columns ||
            required.scales_per_row > capacity_.scales_per_row ||
            (format != Type::FP8 && format != Type::I8))
            throw std::invalid_argument("projection exceeds initialized quantization workspace");
        auto shape = quantization_scale_shape(rows, columns, layout.activation);
        return {values_.view(rows, columns, format), scales_.view(shape.rows, shape.cols, Type::F32),
                layout.activation, layout.rotation};
    }
};
} // namespace tts
