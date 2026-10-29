#include "engine.h"
#include "log.h"
#include "audio_io.h"
#include "tts/tts.h"
#include <algorithm>
#include <cstring>

struct tts_reference {
    std::shared_ptr<const tts::Reference> impl;
};
struct tts_engine {
    tts::Engine impl;
    tts_engine(const char *path, int d, int b, int c,
               tts::Quantization quantization = tts::Quantization::None)
        : impl(path, d, b, c, 128, quantization) {}
};
static void set_error(char *dst, size_t cap, const char *text) {
    TTS_LOG(TTS_LOG_ERROR, "api", "{}", text);
    if (dst && cap) {
        std::strncpy(dst, text, cap - 1);
        dst[cap - 1] = 0;
    }
}
extern "C" {
tts_options tts_default_options() {
    return {TTS_ABI_VERSION, 750, 50, .9f, 1.f, 42, 1};
}
const char *tts_version() {
    return "tts.cpp/0.4.0 abi=1,2 quantization=fp8,int8";
}
tts_engine *tts_create(const char *path, int32_t device, int32_t batch, int32_t context,
                       char *error, size_t cap) {
    return tts_create_with_quantization(path, device, batch, context, nullptr, error, cap);
}
tts_engine *tts_create_with_quantization(const char *path, int32_t device, int32_t batch,
                                         int32_t context, const char *quantization, char *error,
                                         size_t cap) {
    try {
        if (!path || !path[0])
            throw std::runtime_error("model path is required");
        return new tts_engine(path, device, batch, context,
                              tts::quantization_from_string(quantization ? quantization : ""));
    } catch (const std::exception &e) {
        set_error(error, cap, e.what());
        return nullptr;
    } catch (...) {
        set_error(error, cap, "unknown initialization failure");
        return nullptr;
    }
}
void tts_destroy(tts_engine *e) {
    if (tts::log::in_callback()) return;
    delete e;
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
    try {
        if (!e || !out)
            throw std::invalid_argument("null memory stats argument");
        auto stats = e->impl.memory_stats();
        *out = {stats.used_bytes, stats.reserved_bytes, stats.peak_used_bytes,
                stats.peak_reserved_bytes};
        return 0;
    } catch (const std::exception &ex) {
        set_error(error, cap, ex.what());
        return -1;
    } catch (...) {
        set_error(error, cap, "memory stats failed");
        return -1;
    }
}
int tts_generate(tts_engine *e, const tts_request *req, size_t n, const tts_options *options,
                 tts_audio_callback cb, void *user, tts_stats *stats, char *error, size_t cap) {
    try {
        if (!e || !req || !n || !cb || !stats)
            throw std::runtime_error("null argument or empty batch");
        if (n > size_t(e->impl.batch_capacity()))
            throw std::runtime_error("batch exceeds engine capacity");
        auto o = options ? *options : tts_default_options();
        if (o.abi_version != TTS_ABI_VERSION)
            throw std::invalid_argument("unsupported options ABI");
        if (o.use_cuda_graph != 0 && o.use_cuda_graph != 1)
            throw std::invalid_argument("use_cuda_graph must be 0 or 1");
        std::vector<tts::Request> requests;
        for (size_t i = 0; i < n; ++i) {
            if (!req[i].text)
                throw std::invalid_argument("text is required");
            requests.push_back(
                {req[i].text, req[i].instruction ? req[i].instruction : "", "S0", nullptr});
        }
        tts::GenerationOptions native;
        native.max_frames = o.max_frames;
        native.top_k = o.top_k;
        native.temperature = o.temperature;
        native.cfg_scale = o.cfg_scale;
        native.seed = o.seed;
        native.use_cuda_graph = bool(o.use_cuda_graph);
        native.depth_temperature = o.temperature;
        native.depth_top_k = o.top_k;
        native.repetition_penalty = 1.f;
        tts::Stats result;
        int status = e->impl.generate(
            requests, native,
            [&](int i, const int16_t *pcm, size_t size) { return cb(user, i, pcm, size) == 0; },
            result);
        *stats = {result.prefill_ms,    result.first_audio_ms, result.total_ms,
                  result.audio_seconds, result.frames,         result.truncated};
        return status;
    } catch (const std::exception &ex) {
        set_error(error, cap, ex.what());
        return -1;
    } catch (...) {
        set_error(error, cap, "unknown inference failure");
        return -1;
    }
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
tts_options_v2 tts_default_options_v2() {
    return {2, 750, 50, .9f, 1.f, 42, 1, 1.f, 1.1f, 1, 50, .9f, 1.f, 1, 0, 1.f, 1.f};
}
tts_reference *tts_reference_from_audio(tts_engine *e, const void *bytes, size_t size,
                                        const char *text, char *error, size_t cap) {
    try {
        if (!e)
            throw std::invalid_argument("null engine");
        auto t = transcript(text);
        return reference(
            e->impl.encode_audio(tts::read_audio(bytes, size, e->impl.max_reference_samples(),
                                                 e->impl.audio_format().sample_rate)),
            std::move(t), e->impl.audio_format().codebooks, e->impl.audio_format().codebook_size);
    } catch (const std::exception &ex) {
        set_error(error, cap, ex.what());
        return nullptr;
    } catch (...) {
        set_error(error, cap, "reference encoding failed");
        return nullptr;
    }
}
tts_reference *tts_reference_from_pcm(tts_engine *e, const float *pcm, size_t frames, int32_t sr,
                                      int32_t channels, const char *text, char *error, size_t cap) {
    try {
        if (!e)
            throw std::invalid_argument("null engine");
        auto t = transcript(text);
        return reference(e->impl.encode_audio(tts::resample_audio(
                             pcm, frames, sr, channels, e->impl.max_reference_samples(),
                             e->impl.audio_format().sample_rate)),
                         std::move(t), e->impl.audio_format().codebooks,
                         e->impl.audio_format().codebook_size);
    } catch (const std::exception &ex) {
        set_error(error, cap, ex.what());
        return nullptr;
    } catch (...) {
        set_error(error, cap, "reference encoding failed");
        return nullptr;
    }
}
tts_reference *tts_reference_from_codes(const int32_t *codes, size_t frames, int32_t codebooks,
                                        int32_t entries, const char *text, char *error,
                                        size_t cap) {
    try {
        auto t = transcript(text);
        if (!codes || !frames || codebooks < 1 || entries < 1 ||
            frames > SIZE_MAX / sizeof(int) / size_t(codebooks))
            throw std::invalid_argument("invalid reference code count");
        std::vector<int> c(codes, codes + frames * codebooks);
        return reference(std::move(c), std::move(t), codebooks, entries);
    } catch (const std::exception &ex) {
        set_error(error, cap, ex.what());
        return nullptr;
    } catch (...) {
        set_error(error, cap, "reference code import failed");
        return nullptr;
    }
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
int tts_generate_v2(tts_engine *e, const tts_request_v2 *req, size_t n,
                    const tts_options_v2 *options, tts_audio_callback cb, void *user,
                    tts_stats *stats, char *error, size_t cap) {
    try {
        if (!e || !req || !n || !cb || !stats)
            throw std::invalid_argument("null argument or empty batch");
        if (n > size_t(e->impl.batch_capacity()))
            throw std::invalid_argument("batch exceeds engine capacity");
        auto o = options ? *options : tts_default_options_v2();
        if (o.abi_version != 2)
            throw std::invalid_argument("unsupported options ABI");
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
        std::vector<tts::Request> requests;
        for (size_t i = 0; i < n; i++) {
            if (!req[i].text)
                throw std::invalid_argument("text is required");
            requests.push_back({req[i].text, req[i].instruction ? req[i].instruction : "",
                                req[i].speaker ? req[i].speaker : "S0",
                                req[i].reference ? req[i].reference->impl : nullptr});
        }
        tts::Stats r;
        int status = e->impl.generate(
            requests, native,
            [&](int i, const int16_t *pcm, size_t count) { return cb(user, i, pcm, count) == 0; },
            r);
        *stats = {r.prefill_ms,    r.first_audio_ms, r.total_ms,
                  r.audio_seconds, r.frames,         r.truncated};
        return status;
    } catch (const std::exception &ex) {
        set_error(error, cap, ex.what());
        return -1;
    } catch (...) {
        set_error(error, cap, "unknown inference failure");
        return -1;
    }
}
}
