#pragma once
#include <stdexcept>
#include <string>

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
} // namespace tts
