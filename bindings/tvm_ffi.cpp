#include "audio_io.h"
#include "engine.h"
#include "log.h"
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/extra/stl.h>
#include <tvm/ffi/reflection/registry.h>
#include <mutex>
#include <cstring>

namespace tts::python {
namespace ffi = tvm::ffi;
using Values = ffi::Map<ffi::String, ffi::Any>;

struct EngineHandle : ffi::Object {
    std::unique_ptr<Engine> engine;
    static constexpr bool _type_mutable = true;
    TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tts.EngineHandle", EngineHandle, ffi::Object);
    EngineHandle(std::string path, int device, int batch, int context, std::string quantization)
        : engine(std::make_unique<Engine>(path, device, batch, context, 128,
                                          quantization_from_string(quantization))) {}
    Engine &get() const {
        if (!engine)
            throw std::runtime_error("engine is closed");
        return *engine;
    }
};
struct ReferenceHandle : ffi::Object {
    std::shared_ptr<const Reference> reference;
    TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tts.ReferenceHandle", ReferenceHandle, ffi::Object);
    ReferenceHandle(std::vector<int> codes, std::string text, int codebooks, int entries)
        : reference(std::make_shared<const Reference>(
              Reference{std::move(text), std::move(codes), codebooks, entries})) {}
};
struct Options : ffi::Object, GenerationOptions {
    // FFI scalar integers are signed. Keep the RNG seed's full bit pattern.
    int64_t seed_bits = static_cast<int64_t>(GenerationOptions::seed);
    static constexpr bool _type_mutable = true;
    TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tts.Options", Options, ffi::Object);
};

TVM_FFI_STATIC_INIT_BLOCK() {
    ffi::reflection::ObjectDef<EngineHandle>();
    ffi::reflection::ObjectDef<ReferenceHandle>();
    auto fields = ffi::reflection::ObjectDef<Options>();
#define FIELD(name) fields.def_rw(#name, &Options::name)
    FIELD(max_frames);
    FIELD(top_k);
    FIELD(temperature);
    FIELD(cfg_scale);
    fields.def_rw("seed", &Options::seed_bits);
    FIELD(use_cuda_graph);
    FIELD(top_p);
    FIELD(repetition_penalty);
    FIELD(do_sample);
    FIELD(depth_top_k);
    FIELD(depth_temperature);
    FIELD(depth_top_p);
    FIELD(depth_do_sample);
    FIELD(dual_cfg);
    FIELD(cfg_scale_ref);
    FIELD(cfg_scale_ins);
#undef FIELD
}

ffi::ObjectRef create(std::string path, int device, int batch, int context,
                      std::string quantization) {
    try {
        return ffi::ObjectRef(
            ffi::make_object<EngineHandle>(path, device, batch, context, quantization));
    } catch (const std::exception &e) {
        TTS_LOG(TTS_LOG_ERROR, "api", "{}", e.what());
        throw;
    }
}

ffi::ObjectRef reference(EngineHandle *handle, ffi::Bytes audio, std::string text, int rate,
                         int channels) {
    auto &engine = handle->get();
    auto format = engine.audio_format();
    std::vector<float> mono;
    if (rate == 0) {
        mono = read_audio(audio.data(), audio.size(), engine.max_reference_samples(),
                          format.sample_rate);
    } else {
        if (channels < 1 || audio.size() % (sizeof(float) * size_t(channels)))
            throw std::invalid_argument("PCM must contain complete interleaved frames");
        // Host input preparation only; aligned float storage belongs to this call.
        std::vector<float> pcm(audio.size() / sizeof(float));
        std::memcpy(pcm.data(), audio.data(), audio.size());
        mono = resample_audio(pcm.data(), pcm.size() / channels, rate, channels,
                              engine.max_reference_samples(), format.sample_rate);
    }
    return ffi::ObjectRef(ffi::make_object<ReferenceHandle>(
        engine.encode_audio(mono), text, format.codebooks, format.codebook_size));
}

Values generate(EngineHandle *handle, ffi::Array<Values> input, const Options *options,
                ffi::Function deliver) {
    std::vector<Request> requests;
    requests.reserve(input.size());
    for (const auto &r : input) {
        auto ref = r.at("reference");
        requests.push_back(
            {r.at("text").cast<std::string>(), r.at("instruction").cast<std::string>(),
             r.at("speaker").cast<std::string>(),
             ref == nullptr ? nullptr : ref.cast<const ReferenceHandle *>()->reference});
    }
    // Snapshot options before releasing control to Python callbacks.
    GenerationOptions snapshot = *options;
    snapshot.seed = static_cast<uint64_t>(options->seed_bits);
    Stats stats;
    int status = handle->get().generate(
        requests, snapshot,
        [&](int lane, const int16_t *pcm, size_t count) {
            return deliver(lane,
                           ffi::Bytes(reinterpret_cast<const char *>(pcm), count * sizeof(*pcm)))
                .cast<bool>();
        },
        stats);
    return {{"prefill_ms", stats.prefill_ms}, {"first_audio_ms", stats.first_audio_ms},
            {"total_ms", stats.total_ms},     {"audio_seconds", stats.audio_seconds},
            {"frames", stats.frames},         {"truncated", stats.truncated},
            {"cancelled", status == 1}};
}

// Native log configuration serializes emission/replacement. Keep the previous
// callable alive until replacement completes, including in-flight callbacks.
void configure_logging(std::string level, std::string file, bool json,
                       ffi::Optional<ffi::Function> callback) {
    log::check_not_callback();
    static std::mutex mutex;
    static std::unique_ptr<ffi::Function> current;
    auto next = callback ? std::make_unique<ffi::Function>(*callback) : nullptr;
    tts_log_config config{log::parse_level(level), file.empty() ? nullptr : file.c_str(), int(json),
                          nullptr, next.get()};
    if (next) {
        config.callback = [](void *user, const tts_log_record *r) {
            (*static_cast<ffi::Function *>(user))(int(r->level), r->timestamp_us, r->engine_id,
                                                  r->request_id, ffi::String(r->component),
                                                  ffi::String(r->message));
        };
    }
    std::lock_guard<std::mutex> lock(mutex);
    log::configure(config);
    current = std::move(next);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(create, create);
void close(EngineHandle *handle) {
    log::check_not_callback();
    handle->engine.reset();
}
std::string version() {
    return tts_version();
}
ffi::ObjectRef default_options() {
    return ffi::ObjectRef(ffi::make_object<Options>());
}
ffi::ObjectRef copy_options(const Options *options) {
    return ffi::ObjectRef(ffi::make_object<Options>(*options));
}
Values audio_format(const EngineHandle *handle) {
    auto f = handle->get().audio_format();
    return {{"sample_rate", f.sample_rate},
            {"samples_per_frame", f.samples_per_frame},
            {"codebooks", f.codebooks},
            {"codebook_size", f.codebook_size}};
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(audio_format, audio_format);
Values memory_stats(const EngineHandle *handle) {
    auto s = handle->get().memory_stats();
    return {{"used_bytes", s.used_bytes},
            {"reserved_bytes", s.reserved_bytes},
            {"peak_used_bytes", s.peak_used_bytes},
            {"peak_reserved_bytes", s.peak_reserved_bytes}};
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(prepare_reference, reference);
ffi::ObjectRef reference_from_codes(std::vector<int> codes, std::string text, int codebooks,
                                    int entries) {
    return ffi::ObjectRef(
        ffi::make_object<ReferenceHandle>(std::move(codes), std::move(text), codebooks, entries));
}
std::vector<int> reference_codes(const ReferenceHandle *handle) {
    return handle->reference->codes;
}
size_t reference_frames(const ReferenceHandle *handle) {
    return handle->reference->frames();
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(close, close);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(version, version);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(default_options, default_options);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(copy_options, copy_options);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(memory_stats, memory_stats);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(reference_from_codes, reference_from_codes);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(reference_codes, reference_codes);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(reference_frames, reference_frames);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(generate, generate);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(configure_logging, configure_logging);
} // namespace tts::python
