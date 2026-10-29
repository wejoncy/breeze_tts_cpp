#pragma once
#include <cstdint>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>
namespace tts {
struct GenerationOptions {
    // Native parameters; public API adapters translate each field explicitly.
    int max_frames = 750, top_k = 50;
    float temperature = .9f, cfg_scale = 1.f;
    uint64_t seed = 42;
    bool use_cuda_graph = true;
    float top_p = 1.f, repetition_penalty = 1.1f;
    bool do_sample = true;
    int depth_top_k = 50;
    float depth_temperature = .9f, depth_top_p = 1.f;
    bool depth_do_sample = true;
    bool dual_cfg = false;
    float cfg_scale_ref = 1.f, cfg_scale_ins = 1.f;
};
// CPU-owned immutable codes; no CUDA address or request state is retained.
struct Reference {
    std::string text;
    std::vector<int> codes;
};
struct Request {
    std::string text, instruction;
    std::string speaker = "S0";
    std::shared_ptr<const Reference> reference;
};
struct Stats {
    double prefill_ms = 0, first_audio_ms = 0, total_ms = 0, audio_seconds = 0;
    int frames = 0, truncated = 0;
};
using AudioCallback = std::function<bool(int, const int16_t *, size_t)>;
} // namespace tts
