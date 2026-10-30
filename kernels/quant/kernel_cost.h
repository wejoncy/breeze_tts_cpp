#pragma once
#include "kernels/quant/quant_kernel_catalog.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace tts {
// Device rates are in operations or bytes per microsecond, per SM for compute.
// This policy is CPU-only: no execution, calibration, allocation or model identity.
struct ChipModel {
    int sms = 1;
    double memory_bytes_per_us = 1;
    double tensor_ops_per_us = 1;
    double scalar_ops_per_us = 1;
    double kernel_us = 1; // Graph kernel dispatch latency estimate, independent of shape.
    double warp_tensor_ops_per_us = 0;
    double warp_instructions_per_us = 1;
};
inline double tiles(int size, int tile) {
    return (size + tile - 1) / tile;
}
inline double linear_cost(KernelProblem p, LinearPath path, const ChipModel &chip) {
    bool scalar = path.kind == LinearKind::Gemv;
    int tm = path.kind == LinearKind::Lt ? 64 : path.tokens;
    int tn = path.kind == LinearKind::Lt ? 128 : path.channels;
    double mt = tiles(p.rows, tm), nt = tiles(p.channels, tn);
    double blocks = mt * nt * (scalar ? 1 : path.splits);
    double active = std::min<double>(chip.sms, blocks);
    double waves = std::ceil(blocks / chip.sms);
    double utilization = blocks / (waves * chip.sms);
    double operations = 2. * mt * tm * nt * tn * p.reduction;
    bool warp_mma = path.kind == LinearKind::Sm80;
    double rate = scalar                                        ? chip.scalar_ops_per_us
                  : warp_mma && chip.warp_tensor_ops_per_us > 0 ? chip.warp_tensor_ops_per_us
                                                                : chip.tensor_ops_per_us;
    int stage_k = warp_mma && tm == 128 ? 64 : 128;
    int stages = warp_mma ? 3 : 4;
    double steps = std::max(1., tiles(p.reduction, stage_k * path.splits));
    // Match the compiled mainloops. Short K partitions expose pipeline fill/drain.
    double pipeline = scalar ? 1 : 1 + (stages - 1.) / steps;
    double compute = operations / (rate * chip.sms * utilization) * pipeline;
    double bytes = double(p.channels) * p.reduction * mt + double(p.rows) * p.reduction * nt +
                   2. * p.rows * p.channels;
    double memory = bytes / (chip.memory_bytes_per_us * active / chip.sms);
    double cost = chip.kernel_us + std::max(compute, memory);
    if (!scalar && path.kind != LinearKind::Lt) {
        // SM80 issues per-warp 16-byte copies and ldmatrix operand loads; SM90
        // issues two bulk TMA loads and WGMMA reads shared operands directly.
        // Count operand movement instructions as well as bytes: bandwidth-only
        // estimates otherwise make these distinct implementations identical.
        double instructions = warp_mma ? 2. * (tm + tn) * stage_k / (32 * 16) : 2.;
        cost += blocks * steps * instructions / (chip.warp_instructions_per_us * active);
    }
    if (path.splits > 1 && !scalar)
        cost += chip.kernel_us + 8. * path.splits * p.rows * p.channels / chip.memory_bytes_per_us;
    return cost;
}
inline LinearPath select_linear(KernelProblem p, KernelLimits limits, Type format,
                                const ChipModel &chip, bool library_compatible = true) {
    auto paths = quant_linear_candidates(p, limits, format);
    double best = std::numeric_limits<double>::infinity();
    LinearPath selected;
    for (auto path : paths) {
        if (path.kind == LinearKind::Lt && !library_compatible)
            continue;
        double cost = linear_cost(p, path, chip);
        if (cost < best) {
            best = cost;
            selected = path;
        }
    }
    if (!std::isfinite(best))
        throw std::invalid_argument("no supported quantized GEMM implementation");
    return selected;
}
inline int select_projection(KernelProblem p, LinearPath linear, const ChipModel &chip) {
    double best = linear_cost(p, linear, chip) + chip.kernel_us +
                  (3. * p.rows * p.reduction + 4. * p.rows) / chip.memory_bytes_per_us;
    int selected = 0;
    for (int tile : quant_projection_candidates(p)) {
        if (!tile)
            continue;
        LinearPath fused{LinearKind::Gemv, tile, 8, 1};
        double repeated_input = 2. * p.rows * p.reduction * tiles(p.channels, 8);
        double quant_ops = 4. * p.rows * p.reduction * tiles(p.channels, 8);
        double active = std::min<double>(chip.sms, tiles(p.channels, 8) * tiles(p.rows, tile));
        double cost =
            linear_cost(p, fused, chip) + std::max(repeated_input / chip.memory_bytes_per_us,
                                                   quant_ops / (chip.scalar_ops_per_us * active));
        if (cost < best) {
            best = cost;
            selected = tile;
        }
    }
    return selected;
}
} // namespace tts
