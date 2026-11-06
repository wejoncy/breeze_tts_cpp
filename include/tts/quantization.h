#pragma once
#include <stdexcept>
#include <string>

namespace tts {
enum class Quantization { None, FP8, INT8, GroupINT8, GroupFP8 };
inline const char *quantization_name(Quantization quantization) {
    switch (quantization) {
    case Quantization::None: return "none";
    case Quantization::FP8: return "fp8";
    case Quantization::INT8: return "int8";
    case Quantization::GroupINT8: return "group_int8";
    case Quantization::GroupFP8: return "group_fp8";
    }
    throw std::invalid_argument("invalid quantization mode");
}
inline Quantization quantization_from_string(const std::string &name) {
    if (name.empty())
        return Quantization::None;
    if (name == "fp8")
        return Quantization::FP8;
    if (name == "int8")
        return Quantization::INT8;
    if (name == "group_int8")
        return Quantization::GroupINT8;
    if (name == "group_fp8")
        return Quantization::GroupFP8;
    throw std::invalid_argument("quantization must be int8, fp8, group_int8 or group_fp8");
}
} // namespace tts
