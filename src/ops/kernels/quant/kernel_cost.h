#pragma once
#include "ops/kernels/quant/quant_kernel_catalog.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace tts {
// Device rate estimates are in operations or bytes per microsecond, per SM for compute.
// This policy is CPU-only: no execution, calibration, allocation or model identity.
struct ChipModel {
    int sms = 1;
    double memory_bytes_per_us = 1;
    double tensor_ops_per_us = 1;
    double scalar_ops_per_us = 1;
    double kernel_us = 1; // Graph kernel dispatch latency estimate, independent of shape.
    double warp_tensor_ops_per_us = 0;
    double warp_instructions_per_us = 1;
    size_t cache_bytes = 0;
    double cache_bytes_per_us = 0;
};
inline double tiles(int size, int tile) {
    return (size + tile - 1) / tile;
}
inline double traffic_cost(KernelProblem p, double bytes, const ChipModel &chip) {
    double memory = bytes / chip.memory_bytes_per_us;
    double working_set = (double(p.channels) + p.rows) * p.reduction +
                         2. * p.rows * p.channels;
    // A fitting operand/output set can serve tiled rereads from L2.
    if (chip.cache_bytes_per_us > 0 && working_set <= chip.cache_bytes)
        memory = std::max(working_set / chip.memory_bytes_per_us,
                          bytes / chip.cache_bytes_per_us);
    return memory;
}
// Source issue lower bound for the existing eight-warp GEMV algorithm.
// Exposed memory/barrier latency and compiler setup can only increase it.
inline double gemv_issue_work(KernelProblem p, LinearPath path, Type format) {
    const int tokens = path.tokens, cols = path.channels, splits = path.splits;
    const int warps = cols * splits;
    const int groups = int(tiles(p.reduction, 128)); // 32 lanes times four packed bytes.
    double chunks = 0;
    for (int split = 0; split < splits; ++split) {
        const int count = std::max(0, (groups - split + splits - 1) / splits);
        chunks += count / 4 + count % 4; // Existing four-word block and guarded tail.
    }
    // Staging: global load, shared store, index advance, comparison, branch.
    double issue = tiles(tokens * p.reduction, 128) * 5;
    // CUDA FP8 float4: high-half extract, two pair decodes, four float converts.
    // Weight decoding is reused across the token tile; INT8 dot is one DP4A.
    const double dot = format == Type::I8 ? tokens
        : 4 * tokens + (1 + 2 + 4) * (1 + tokens);
    // Mainloop weight load, shared input loads and ordered packed dots/decodes.
    issue += cols * groups * (1 + tokens + dot);
    // 64-bit weight pointer, token shared pointers, word advance, compare, branch.
    issue += cols * chunks * (2 + tokens + 1 + 1 + 1);
    // Native integer reduction/result transfer, or original five shuffle/add pairs.
    issue += warps * tokens * (format == Type::I8 ? 2 : 10);
    // Per output: two scale loads, two multiplies, cast/store, and INT8 conversion.
    issue += cols * tokens * (6 + (format == Type::I8));
    issue += warps; // The staging CTA barrier is issued by each warp.
    if (splits > 1) {
        issue += warps * tokens + cols * tokens * splits; // Partial store/load.
        issue += cols * tokens * (splits - (format == Type::I8)); // Ordered partial sums.
        issue += warps; // Second CTA barrier, not a second kernel launch.
    }
    return issue;
}
inline double library_cost(KernelProblem p, const ChipModel &chip, LibraryWork work) {
    // The heuristic reports effective utilization, not necessarily a physical
    // CTA count. Unknown estimates conservatively cover one SM.
    double blocks = std::isfinite(work.effective_waves)
                        ? std::max(1., std::round(work.effective_waves * chip.sms))
                        : 1.;
    double active = std::min<double>(chip.sms, blocks);
    double utilization = blocks / (std::ceil(blocks / chip.sms) * chip.sms);
    // Minimum operand traffic for this effective tile count: minimize
    // N*mt + M*nt subject to mt*nt=blocks and mt,nt>=1. No assumed Lt tile.
    double mt = std::clamp(std::sqrt(blocks * p.rows / p.channels), 1., blocks);
    double nt = blocks / mt;
    double bytes = (p.channels * mt + p.rows * nt) * p.reduction +
                   2. * p.rows * p.channels;
    double memory = traffic_cost(p, bytes, chip) * chip.sms / active;
    double compute = 2. * p.rows * p.channels * p.reduction /
                     (chip.tensor_ops_per_us * chip.sms * utilization);
    return work.kernel_launches * chip.kernel_us + std::max(compute, memory);
}
inline double linear_cost(KernelProblem p, LinearPath path, Type format, const ChipModel &chip,
                          LibraryWork library_work = {}) {
    if (path.kind == LinearKind::Lt)
        return library_cost(p, chip, library_work);
    bool scalar = path.kind == LinearKind::Gemv;
    int tm = path.tokens, tn = path.channels;
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
    // Each FP8 quad needs six instructions to decode the shared weight;
    // each token adds six input conversions and four FMAs. The compiler
    // reuses the decoded weight across the token tile.
    if (scalar && format == Type::FP8)
        rate *= 4. * tm / (6. + 10. * tm);
    int stage_k = warp_mma && tm == 128 ? 64 : 128;
    int stages = warp_mma ? 3 : 4;
    double steps = std::max(1., tiles(p.reduction, stage_k * path.splits));
    // Match the compiled mainloops. Short K partitions expose pipeline fill/drain.
    double pipeline = scalar ? 1 : 1 + (stages - 1.) / steps;
    double compute = operations / (rate * chip.sms * utilization) * pipeline;
    if (scalar) {
        // A CTA stays on one SM: the busiest SM must issue whole CTA work,
        // including a partial final grid wave, just as the arithmetic bound.
        compute = std::max(compute, waves * gemv_issue_work(p, path, format) /
                                       chip.warp_instructions_per_us);
    }
    // A single consumer warp-group drains each K128 partial before adding it
    // to FP32 registers. These warp FADD instructions serialize with its MMA.
    if (format == Type::FP8 && path.kind == LinearKind::Hopper)
        compute += blocks * steps * tm * tn / 32. /
                   (chip.warp_instructions_per_us * active);
    double bytes = double(p.channels) * p.reduction * mt + double(p.rows) * p.reduction * nt +
                   2. * p.rows * p.channels;
    double memory = scalar ? bytes / chip.memory_bytes_per_us : traffic_cost(p, bytes, chip);
    double coverage = active / chip.sms;
    if (scalar) {
        // Eight warps per CTA. Four schedulers need four independent warps
        // each to hide the four-cycle packed-dot dependency. A nearly full
        // grid of one CTA per SM does not have the same coverage as two CTAs.
        coverage = std::min(1., blocks * 8 / (chip.sms * 16.));
    }
    memory /= coverage;
    double cost = chip.kernel_us + std::max(compute, memory);
    if (!scalar) {
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
// Operator work is passed as plain data; this CPU policy knows no activation enum.
struct LinearFinishWork { double instructions_per_output = 0; };
inline double linear_pipeline_cost(KernelProblem p, LinearPath path, Type format,
                                   const ChipModel &chip, LinearFinishWork finish, bool fused,
                                   LibraryWork library = {}, KernelResources resources = {}) {
    double elements = double(p.rows) * p.channels;
    double active = chip.sms;
    if (fused && path.kind != LinearKind::Lt) {
        if (resources.resident_ctas < 1) return std::numeric_limits<double>::infinity();
        double blocks = tiles(p.rows, path.tokens) * tiles(p.channels, path.channels);
        elements = blocks * path.tokens * path.channels;
        active = std::min<double>(chip.sms, blocks);
    }
    double activation = elements * finish.instructions_per_output /
        (32. * chip.warp_instructions_per_us * active);
    double quantize = chip.kernel_us +
        (3. * p.rows * p.channels + sizeof(float) * p.rows) / chip.memory_bytes_per_us;
    double gemm = linear_cost(p, path, format, chip, library);
    if (!fused) return gemm + activation + quantize;
    if (path.kind == LinearKind::Lt) return std::max(gemm, activation) + quantize;
    // Native consumers finish after their own mainloop; overlap is possible
    // across resident CTAs. A single resident CTA retains the serial sum.
    return std::max(gemm, activation) +
           std::min(gemm, activation) / resources.resident_ctas + quantize;
}
inline LinearPath select_linear(KernelProblem p, KernelLimits limits, Type format,
                                const ChipModel &chip, bool library_compatible = true,
                                LibraryWork library_work = {}) {
    auto paths = quant_linear_candidates(p, limits, format);
    double best = std::numeric_limits<double>::infinity();
    LinearPath selected;
    for (auto path : paths) {
        if (path.kind == LinearKind::Lt && !library_compatible)
            continue;
        double cost = linear_cost(p, path, format, chip, library_work);
        if (cost < best) {
            best = cost;
            selected = path;
        }
    }
    if (!std::isfinite(best))
        throw std::invalid_argument("no supported quantized GEMM implementation");
    return selected;
}
} // namespace tts
