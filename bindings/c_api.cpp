#include "tts/engine.h"
#include "support/log.h"
#include "io/audio_io.h"
#include "tts/tts.h"
#include <algorithm>
#include <cstring>
#include <type_traits>

struct tts_reference {
    std::shared_ptr<const tts::Reference> impl;
};
struct tts_engine {
    tts::Engine impl;
    tts_engine(const char *path, int d, int b, int c,
               tts::Quantization quantization = tts::Quantization::None)
        : impl(path, d, b, c, 128, quantization) {}
};
static void set_error(char *dst, size_t cap, const char *text) noexcept {
    try {
        TTS_LOG(TTS_LOG_ERROR, "api", "{}", text);
    } catch (...) {
    }
    if (dst && cap) {
        std::strncpy(dst, text, cap - 1);
        dst[cap - 1] = 0;
    }
}
template <class Function>
auto api_call(char *error, size_t capacity, const char *fallback, const Function &function)
    -> decltype(function()) {
    try {
        return function();
    } catch (const std::exception &exception) {
        set_error(error, capacity, exception.what());
    } catch (...) {
        set_error(error, capacity, fallback);
    }
    if constexpr (std::is_pointer_v<decltype(function())>) {
        return nullptr;
    } else {
        return -1;
    }
}
extern "C" {
const char *tts_version() {
    static const std::string version = "tts.cpp/0.4.0 abi=" + std::to_string(TTS_ABI_VERSION) +
                                       " quantization=int8,fp8,group_int8,group_fp8";
    return version.c_str();
}
tts_engine *tts_create(const char *path, int32_t device, int32_t batch, int32_t context,
                       const char *quantization, char *error, size_t cap) {
    return api_call(error, cap, "unknown initialization failure", [&] {
        if (!path || !path[0])
            throw std::runtime_error("model path is required");
        return new tts_engine(path, device, batch, context,
                              tts::quantization_from_string(quantization ? quantization : ""));
    });
}
void tts_destroy(tts_engine *e) {
    if (tts::log::in_callback())
        return;
    delete e;
}
int tts_run_async(tts_engine *e, char *error, size_t cap) {
    return api_call(error, cap, "engine startup failed", [&] {
        if (!e)
            throw std::invalid_argument("null engine");
        e->impl.run_async();
        return 0;
    });
}
int tts_shutdown(tts_engine *e, char *error, size_t cap) {
    return api_call(error, cap, "engine shutdown failed", [&] {
        if (!e)
            throw std::invalid_argument("null engine");
        e->impl.shutdown();
        return 0;
    });
}
int tts_get_audio_format(const tts_engine *e, tts_audio_format *out, char *error, size_t cap) {
    if (!e || !out) {
        set_error(error, cap, "null audio format argument");
        return -1;
    }
    auto f = e->impl.audio_format();
    *out = {f.sample_rate, f.samples_per_frame, f.codebooks, f.codebook_size};
    return 0;
}
int tts_get_memory_stats(tts_engine *e, tts_memory_stats *out, char *error, size_t cap) {
    return api_call(error, cap, "memory stats failed", [&] {
        if (!e || !out)
            throw std::invalid_argument("null memory stats argument");
        auto stats = e->impl.memory_stats();
        *out = {stats.used_bytes, stats.reserved_bytes, stats.peak_used_bytes,
                stats.peak_reserved_bytes};
        return 0;
    });
}
}

namespace {
std::string transcript(const char *text) {
    if (!text || !text[0] || std::strlen(text) > 65536)
        throw std::invalid_argument("reference transcript is required (1..65536 bytes)");
    return text;
}
tts_reference *reference(std::vector<int> codes, std::string text, int codebooks, int entries) {
    return new tts_reference{std::make_shared<const tts::Reference>(
        tts::Reference{std::move(text), std::move(codes), codebooks, entries})};
}
} // namespace
extern "C" {
tts_options tts_default_options() {
    const tts::GenerationOptions o;
    return {TTS_ABI_VERSION,
            o.max_frames,
            o.top_k,
            o.temperature,
            o.cfg_scale,
            o.seed,
            int(o.use_cuda_graph),
            o.top_p,
            o.repetition_penalty,
            int(o.do_sample),
            o.depth_top_k,
            o.depth_temperature,
            o.depth_top_p,
            int(o.depth_do_sample),
            int(o.dual_cfg),
            o.cfg_scale_ref,
            o.cfg_scale_ins,
            o.speed_factor,
            o.sample_steps,
            o.cfg_rate,
            int(o.synthesis_mode),
            o.streaming_chunk_seconds,
            o.streaming_lookahead_seconds,
            o.streaming_overlap_seconds,
            o.semantic_lookahead_seconds};
}
tts_reference *tts_reference_from_audio(tts_engine *e, const void *bytes, size_t size,
                                        const char *text, const char *language, char *error, size_t cap) {
    return api_call(error, cap, "reference encoding failed", [&] {
        if (!e)
            throw std::invalid_argument("null engine");
        auto t = transcript(text);
        return new tts_reference{e->impl.prepare_reference(
            tts::read_audio(bytes, size, e->impl.max_reference_samples(),
                             e->impl.audio_format().sample_rate), std::move(t), language ? language : "")};
    });
}
tts_reference *tts_reference_from_pcm(tts_engine *e, const float *pcm, size_t frames, int32_t sr,
                                      int32_t channels, const char *text, const char *language,
                                      char *error, size_t cap) {
    return api_call(error, cap, "reference encoding failed", [&] {
        if (!e)
            throw std::invalid_argument("null engine");
        auto t = transcript(text);
        return new tts_reference{e->impl.prepare_reference(
            tts::resample_audio(pcm, frames, sr, channels, e->impl.max_reference_samples(),
                                e->impl.audio_format().sample_rate), std::move(t), language ? language : "")};
    });
}
tts_reference *tts_reference_from_codes(const int32_t *codes, size_t frames, int32_t codebooks,
                                        int32_t entries, const char *text, char *error,
                                        size_t cap) {
    return api_call(error, cap, "reference code import failed", [&] {
        auto t = transcript(text);
        if (!codes || !frames || codebooks < 1 || entries < 1 ||
            frames > SIZE_MAX / sizeof(int) / size_t(codebooks))
            throw std::invalid_argument("invalid reference code count");
        std::vector<int> c(codes, codes + frames * codebooks);
        return reference(std::move(c), std::move(t), codebooks, entries);
    });
}
size_t tts_reference_frames(const tts_reference *r) {
    return r ? r->impl->frames() : 0;
}
size_t tts_reference_codes(const tts_reference *r, int32_t *out, size_t cap) {
    if (!r || !out || cap < r->impl->codes.size())
        return 0;
    std::copy(r->impl->codes.begin(), r->impl->codes.end(), out);
    return r->impl->frames();
}
void tts_reference_destroy(tts_reference *r) {
    delete r;
}
int tts_generate(tts_engine *e, const tts_request *req, size_t n, const tts_options *options,
                 tts_audio_callback cb, void *user, tts_stats *stats, char *error, size_t cap) {
    return api_call(error, cap, "unknown inference failure", [&] {
        if (!e || !req || !n || !cb || !stats)
            throw std::invalid_argument("null argument or empty batch");
        if (options && options->abi_version != TTS_ABI_VERSION)
            throw std::invalid_argument("unsupported options ABI");
        auto o = options ? *options : tts_default_options();
        for (int flag : {o.use_cuda_graph, o.do_sample, o.depth_do_sample, o.dual_cfg})
            if (flag != 0 && flag != 1)
                throw std::invalid_argument("boolean options must be 0 or 1");
        tts::GenerationOptions native;
        native.max_frames = o.max_frames;
        native.top_k = o.top_k;
        native.temperature = o.temperature;
        native.cfg_scale = o.cfg_scale;
        native.seed = o.seed;
        native.use_cuda_graph = bool(o.use_cuda_graph);
        native.top_p = o.top_p;
        native.repetition_penalty = o.repetition_penalty;
        native.do_sample = bool(o.do_sample);
        native.depth_top_k = o.depth_top_k;
        native.depth_temperature = o.depth_temperature;
        native.depth_top_p = o.depth_top_p;
        native.depth_do_sample = bool(o.depth_do_sample);
        native.dual_cfg = bool(o.dual_cfg);
        native.cfg_scale_ref = o.cfg_scale_ref;
        native.cfg_scale_ins = o.cfg_scale_ins;
        native.speed_factor = o.speed_factor;
        native.sample_steps = o.sample_steps;
        native.cfg_rate = o.cfg_rate;
        if (o.synthesis_mode < TTS_SYNTHESIS_OFFLINE || o.synthesis_mode > TTS_SYNTHESIS_HYBRID)
            throw std::invalid_argument("invalid synthesis mode");
        native.synthesis_mode = static_cast<tts::SynthesisMode>(o.synthesis_mode);
        native.streaming_chunk_seconds = o.streaming_chunk_seconds;
        native.streaming_lookahead_seconds = o.streaming_lookahead_seconds;
        native.streaming_overlap_seconds = o.streaming_overlap_seconds;
        native.semantic_lookahead_seconds = o.semantic_lookahead_seconds;
        if (!options)
            native = e->impl.default_options();
        std::vector<tts::Input> requests;
        for (size_t i = 0; i < n; i++) {
            if (!req[i].text)
                throw std::invalid_argument("text is required");
            requests.push_back({req[i].text, req[i].instruction ? req[i].instruction : "",
                                req[i].speaker ? req[i].speaker : "S0",
                                req[i].reference ? req[i].reference->impl : nullptr,
                                req[i].language ? req[i].language : "",
                                req[i].text_split_method ? req[i].text_split_method : "",
                                req[i].fragment_interval});
        }
        tts::Stats r;
        int status = e->impl.generate(
            requests, native,
            [&](int i, const int16_t *pcm, size_t count) { return cb(user, i, pcm, count) == 0; },
            r);
        *stats = {r.prefill_ms,    r.first_audio_ms, r.total_ms,
                  r.audio_seconds, r.frames,         r.truncated};
        return status;
    });
}
}
