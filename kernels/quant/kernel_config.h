#pragma once
#include "kernels/quant/kernel_problem.h"
#include <cstddef>
#include <string>
namespace tts {
enum class KernelOperation { ScaledLinear, Bf16InputProjection };
enum class LinearKind { Lt, Gemv, Sm80, Hopper };
struct LinearPath {
    LinearKind kind = LinearKind::Lt;
    int tokens = 1, channels = 0, splits = 1;
};
inline std::string path_name(const LinearPath &p) {
    if (p.kind == LinearKind::Lt)
        return "cublaslt";
    if (p.kind == LinearKind::Gemv)
        return "gemv_tokens" + std::to_string(p.tokens) + "_split" + std::to_string(p.splits);
    return std::string(p.kind == LinearKind::Sm80 ? "sm80_channels" : "hopper_channels") +
           std::to_string(p.channels) + "_tokens" + std::to_string(p.tokens) + "_split" +
           std::to_string(p.splits);
}
struct KernelChoice {
    LinearPath linear;
    int projection_tokens = 0; // zero means quantize + the selected linear
};
} // namespace tts
