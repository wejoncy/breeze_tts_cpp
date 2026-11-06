#pragma once
#include "core/tensor.h"
#include "tts/quantization.h"

namespace tts {
inline Type quantized_type(Quantization quantization) {
    if (quantization == Quantization::FP8 || quantization == Quantization::GroupFP8)
        return Type::FP8;
    if (quantization == Quantization::INT8 || quantization == Quantization::GroupINT8)
        return Type::I8;
    throw std::invalid_argument("BF16 has no quantized storage type");
}
inline const char *quantized_name(Type type) {
    return type == Type::I8 ? "INT8" : "FP8";
}
struct QuantizationGroups {
    int activation = 0, weight = 0, rotation = 0;
};
inline QuantizationGroups quantization_groups(Quantization quantization) {
    if (quantization == Quantization::GroupINT8 || quantization == Quantization::GroupFP8)
        return {128, 128, 128};
    return {};
}
inline TensorShape quantization_scale_shape(int rows, int columns, int group_size) {
    return group_size ? TensorShape{(columns + group_size - 1) / group_size, rows}
                      : TensorShape{rows, 1};
}
// Values are row-major. A zero group size uses one F32 scale per row.
// K groups use [ceil(K / group_size), rows] scales, with rows contiguous;
// this is also cuBLASLt's VEC128_32F layout. No storage/runtime dependency.
// Rotation is semantic metadata: transformed and untransformed values have
// identical shapes. Producers apply it; GEMM only checks that operands match.
struct RowScaledView {
    TensorView values, scales;
    int group_size = 0, rotation_size = 0;
    int groups() const {
        return group_size ? (values.cols + group_size - 1) / group_size : 1;
    }
    void validate() const {
        if ((values.type != Type::FP8 && values.type != Type::I8) || !values.dense() ||
            (rotation_size && (rotation_size != 128 || values.cols % 128)) ||
            (group_size && group_size != 32 && group_size != 64 && group_size != 128) ||
            scales.type != Type::F32 || !scales.dense() ||
            scales.numel() != size_t(values.rows) * groups() ||
            (group_size && (scales.rows != groups() || scales.cols != values.rows)) ||
            values.device != scales.device)
            throw std::invalid_argument("invalid row-scaled quantized tensor");
    }
};
} // namespace tts
