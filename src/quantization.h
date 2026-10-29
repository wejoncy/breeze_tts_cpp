#pragma once
#include "tensor.h"
#include <string>
#include <cstdlib>

namespace tts {
enum class Quantization { None, FP8, INT8 };
inline Quantization quantization_from_string(const std::string &name) {
    if (name.empty())
        return Quantization::None;
    if (name == "fp8")
        return Quantization::FP8;
    if (name == "int8")
        return Quantization::INT8;
    throw std::invalid_argument("quantization must be fp8 or int8");
}
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
inline bool quant_autotune_enabled() {
    const char *v = std::getenv("TTS_QUANT_AUTOTUNE");
    return !v || std::string(v) != "0";
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
