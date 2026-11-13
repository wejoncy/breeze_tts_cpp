#pragma once
#include <cstdint>
#include <utility>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <exception>
namespace tts {
struct RequestOverloaded : std::runtime_error {
    using std::runtime_error::runtime_error;
};
struct AudioFormat {
    int sample_rate, samples_per_frame, codebooks, codebook_size;
};
inline uint64_t request_seed(uint64_t seed, size_t lane) {
    return seed + uint64_t(lane) * 0xd1342543de82ef95ULL;
}
enum class SynthesisMode { Offline, Fixed, Hybrid };
inline const char *synthesis_mode_name(SynthesisMode mode) {
    switch (mode) {
    case SynthesisMode::Offline: return "offline";
    case SynthesisMode::Fixed: return "fixed";
    case SynthesisMode::Hybrid: return "hybrid";
    }
    throw std::invalid_argument("invalid synthesis mode");
}
inline SynthesisMode parse_synthesis_mode(const std::string &name) {
    for (auto mode : {SynthesisMode::Offline, SynthesisMode::Fixed, SynthesisMode::Hybrid})
        if (name == synthesis_mode_name(mode)) return mode;
    throw std::invalid_argument("synthesis_mode must be offline, fixed or hybrid");
}
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
    double speed_factor = 1.0;
    int sample_steps = 0;
    float cfg_rate = -1.f;
    SynthesisMode synthesis_mode = SynthesisMode::Fixed;
    // Zero chunk / negative context values select the model's configured defaults.
    double streaming_chunk_seconds = 0, streaming_lookahead_seconds = -1;
    double streaming_overlap_seconds = -1, semantic_lookahead_seconds = -1;
};
// CPU-owned immutable reference features; no CUDA address or request state is retained.
// Models can attach spectrograms, speaker features or other conditioning without
// exposing their layout to the engine, scheduler or language bindings.
struct ReferenceFeatures {
    virtual ~ReferenceFeatures() = default;
};
struct Reference {
    std::string text;
    std::vector<int> codes;
    int codebooks = 0, codebook_size = 0;
    std::shared_ptr<const ReferenceFeatures> features;
    Reference() = default;
    Reference(std::string transcript, std::shared_ptr<const ReferenceFeatures> conditioning)
        : text(std::move(transcript)), features(std::move(conditioning)) {
        if (!features)
            throw std::invalid_argument("reference conditioning is empty");
    }
    Reference(std::string transcript, std::vector<int> values, int books, int entries)
        : text(std::move(transcript)), codes(std::move(values)), codebooks(books),
          codebook_size(entries) {
        if (codebooks < 1 || codebook_size < 1 || codes.empty() || codes.size() % codebooks ||
            std::any_of(
                codes.begin(), codes.end(), [&](int v) { return v < 0 || v >= codebook_size; }))
            throw std::invalid_argument("reference codes do not match their declared layout");
    }
    size_t frames() const {
        return codebooks > 0 ? codes.size() / codebooks : 0;
    }
};
struct Input {
    std::string text, instruction;
    std::string speaker = "S0";
    std::shared_ptr<const Reference> reference;
    std::string language; // Empty selects the model's default text frontend.
    std::string text_split_method; // Empty preserves a single input.
    double fragment_interval = 0;
};
// One submitted request may contain several input lanes. The caller owns its identity.
struct Request {
    std::string request_id;
    std::vector<Input> inputs;
    GenerationOptions options;
};
struct Stats {
    double total_ms = 0;
    int truncated = 0;
};
struct GenerationFrame {
    std::string request_id;
    std::vector<std::pair<int, std::vector<int16_t>>> chunks;
    Stats stats;
    int batch_size = 0;
    std::exception_ptr error;
    bool done = false, cancelled = false;
};
} // namespace tts
