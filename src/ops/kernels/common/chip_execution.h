#pragma once
namespace tts {
// Native FP32 SIMT width: SM80 has 64 lanes; GA10x and Hopper have 128.
// These are chip execution traits, independent of model and tensor shape.
inline int fp32_lanes_per_sm(int major, int minor) {
    return major == 8 && minor == 0 ? 64 : 128;
}
inline int fp32_latency_warps(int major, int minor) {
    constexpr int dependent_fma_cycles = 4, warp_lanes = 32;
    return dependent_fma_cycles * fp32_lanes_per_sm(major, minor) / warp_lanes;
}
} // namespace tts
