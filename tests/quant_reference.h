#pragma once
#include "quantization.h"
#include <cuda_fp8.h>
#include <algorithm>
#include <cmath>
namespace quant_test {
inline tts::Type storage = tts::Type::FP8;
inline tts::Quantization quantization(int argc, char **argv) {
    auto p = tts::quantization_from_string(argc > 1 ? argv[1] : "fp8");
    storage = tts::quantized_type(p);
    return p;
}
inline float range() {
    return storage == tts::Type::I8 ? 127.f : 448.f;
}
// Independent host oracle: nearest-even symmetric INT8 and byte sign extension.
inline unsigned char encode(float v) {
    if (storage == tts::Type::I8)
        return static_cast<unsigned char>(int(std::nearbyint(std::clamp(v, -127.f, 127.f))));
    return __nv_fp8_e4m3(v).__x;
}
inline float decode(unsigned char byte) {
    if (storage == tts::Type::I8)
        return byte < 128 ? int(byte) : int(byte) - 256;
    __nv_fp8_e4m3 x;
    x.__x = byte;
    return float(x);
}
} // namespace quant_test
