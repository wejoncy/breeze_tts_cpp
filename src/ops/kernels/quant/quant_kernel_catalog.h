#pragma once
#include "ops/kernels/quant/kernel_config.h"
#include "core/quantization.h"
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
    if (p.reduction % 4 == 0) {
        auto gemv = [&](int tokens, int splits) {
            // Supported CUDA devices have a 65535-block grid Y limit. The
            // source-level static array is conservatively included for all splits.
            // shared_bytes excludes the CUDA driver reservation.
            if ((int64_t(p.rows) + tokens - 1) / tokens <= 65535 &&
                size_t(tokens) * (p.reduction + 8 * sizeof(float)) <= limits.shared_bytes)
                paths.push_back({LinearKind::Gemv, tokens, 8 / splits, splits});
        };
        for (int split : {1, 2, 4})
            gemv(1, split);
        for (int tokens : {2, 4})
            gemv(tokens, 1);
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
            for (int channels : {64, 128}) {
                // FP8 keeps WGMMA partials and promoted FP32 accumulators.
                // Their lower bound alone must fit the SM90 limit of 255
                // registers per thread in a 128-thread consumer warp-group.
                int copies = format == Type::FP8 ? 2 : 1;
                if (kind == LinearKind::Hopper && channels * tokens * copies > 255 * 128)
                    continue;
                paths.push_back({kind, tokens, channels, 1});
            }
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
} // namespace tts
