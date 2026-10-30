#pragma once
#include "core/tensor.h"
#include "tts/quantization.h"

namespace tts {
inline Type quantized_type(Quantization quantization) {
    if (quantization == Quantization::FP8)
        return Type::FP8;
    if (quantization == Quantization::INT8)
        return Type::I8;
    throw std::invalid_argument("BF16 has no quantized storage type");
}
inline const char *quantized_name(Type type) {
    return type == Type::I8 ? "INT8" : "FP8";
}
// Values are row-major; scales are F32, one per row. Rows mean tokens for
// activations and output channels for weights. No storage/backend dependency.
struct RowScaledView {
    TensorView values, scales;
    void validate() const {
        if ((values.type != Type::FP8 && values.type != Type::I8) || !values.dense() ||
            scales.type != Type::F32 || !scales.dense() || scales.numel() != size_t(values.rows))
            throw std::invalid_argument("invalid row-scaled quantized tensor");
    }
};
} // namespace tts
