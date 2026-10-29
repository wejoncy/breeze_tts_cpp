#pragma once
#include "kernel_config.h"
#include "quantization.h"
#include <vector>

namespace tts {
struct KernelLimits {
    bool hopper = false;
    size_t shared_bytes = 0, workspace_bytes = 0;
};
// Supported implementation geometries and legality checks live beside dispatch,
// independently of model dimensions and selection/timing/cache policy.
inline std::vector<LinearPath> quant_linear_candidates(KernelProblem p, KernelLimits limits,
                                                       Type format = Type::FP8) {
    std::vector<LinearPath> paths;
    if (format == Type::FP8)
        paths.push_back({});
    if (p.rows <= 8 && p.reduction % 128 == 0 && size_t(p.reduction) * 4 <= limits.shared_bytes) {
        for (int split : {1, 2, 4})
            paths.push_back({LinearKind::Gemv, 1, 8 / split, split});
        if (p.rows > 1)
            for (int tokens : {2, 4})
                paths.push_back({LinearKind::Gemv, tokens, 8, 1});
    }
    for (auto kind : {LinearKind::Sm80, LinearKind::Hopper}) {
        bool supported = kind == LinearKind::Sm80
                             ? format == Type::I8
                             : limits.hopper && (p.channels >= 256 || format == Type::I8);
        if (!supported || p.reduction % 16)
            continue;
        for (int tokens : {16, 32, 64, 128}) {
            if (tokens > 16 && p.rows <= tokens / 2)
                continue;
            for (int channels : {64, 128})
                paths.push_back({kind, tokens, channels, 1});
            for (int channels : {64, 128}) {
                if (kind == LinearKind::Hopper && channels != 64)
                    continue;
                for (int split : {2, 4, 8, 16})
                    if (p.reduction % (split * 16) == 0 &&
                        size_t(split) * p.rows * p.channels * 4 <= limits.workspace_bytes)
                        paths.push_back({kind, tokens, channels, split});
            }
        }
    }
    return paths;
}
inline std::vector<int> quant_projection_candidates(KernelProblem p) {
    if (p.rows <= 32 && (p.reduction == 1024 || p.reduction == 2048))
        return {0, 1, 2, 4, 8};
    return {0};
}
} // namespace tts
