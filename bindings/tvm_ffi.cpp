#include <chrono>
#include "io/audio_io.h"
#include "tts/engine.h"
#include "support/log.h"
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/extra/stl.h>
#include <tvm/ffi/reflection/registry.h>
#include <mutex>
#include <cstring>

namespace tts::python {
namespace ffi = tvm::ffi;
using Values = ffi::Map<ffi::String, ffi::Any>;

template <class F> decltype(auto) with_errors(F &&f) {
    try {
        return f();
    } catch (const RequestOverloaded &error) {
        TVM_FFI_THROW(TTSOverloaded) << error.what();
    }
}

struct EngineHandle : ffi::Object {
    // The FFI argument retains immutable engine storage for the whole call.
    const std::unique_ptr<Engine> engine;
    static constexpr bool _type_mutable = true;
    TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tts.EngineHandle", EngineHandle, ffi::Object);
    EngineHandle(std::string path, int device, int batch, int context, std::string quantization,
                 int64_t runtime_memory_bytes, int64_t memory_limit_bytes)
        : engine(std::make_unique<Engine>(
              path, device, batch, context, 128, quantization_from_string(quantization),
              size_t(runtime_memory_bytes), size_t(memory_limit_bytes))) {}
    Engine *get() const {
        return engine.get();
    }
};
struct ReferenceHandle : ffi::Object {
    std::shared_ptr<const Reference> reference;
    TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tts.ReferenceHandle", ReferenceHandle, ffi::Object);
    explicit ReferenceHandle(std::shared_ptr<const Reference> value)
        : reference(std::move(value)) {}
    ReferenceHandle(std::vector<int> codes, std::string text, int codebooks, int entries)
        : reference(std::make_shared<const Reference>(
              Reference{std::move(text), std::move(codes), codebooks, entries})) {}
};
struct Options : ffi::Object, GenerationOptions {
    // FFI scalar integers are signed. Keep the RNG seed's full bit pattern.
    int64_t seed_bits = static_cast<int64_t>(GenerationOptions::seed);
    std::string synthesis_mode_name = tts::synthesis_mode_name(GenerationOptions::synthesis_mode);
    Options() = default;
    explicit Options(GenerationOptions options) : GenerationOptions(std::move(options)),
        synthesis_mode_name(tts::synthesis_mode_name(GenerationOptions::synthesis_mode)) {}
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
    FIELD(speed_factor);
    FIELD(sample_steps);
    FIELD(cfg_rate);
    fields.def_rw("synthesis_mode", &Options::synthesis_mode_name);
    FIELD(streaming_chunk_seconds);
    FIELD(streaming_lookahead_seconds);
    FIELD(streaming_overlap_seconds);
    FIELD(semantic_lookahead_seconds);
#undef FIELD
}

ffi::ObjectRef create(std::string path, int device, int batch, int context,
                      std::string quantization, int64_t runtime_memory_bytes,
                      int64_t memory_limit_bytes) {
    try {
        if (runtime_memory_bytes < 0 || memory_limit_bytes < 0)
            throw std::invalid_argument("memory budgets must be nonnegative");
        return ffi::ObjectRef(ffi::make_object<EngineHandle>(
            path, device, batch, context, quantization, runtime_memory_bytes, memory_limit_bytes));
    } catch (const std::exception &e) {
        TTS_LOG(TTS_LOG_ERROR, "api", "{}", e.what());
        throw;
    }
}

ffi::ObjectRef reference(EngineHandle *handle, ffi::Bytes audio, std::string text, int rate,
                         int channels, std::string language) {
    auto engine = handle->get();
    auto format = engine->audio_format();
    std::vector<float> mono;
    if (rate == 0) {
        mono = read_audio(audio.data(), audio.size(), engine->max_reference_samples(),
                          format.sample_rate);
    } else {
        if (channels < 1 || audio.size() % (sizeof(float) * size_t(channels)))
            throw std::invalid_argument("PCM must contain complete interleaved frames");
        // Host input preparation only; aligned float storage belongs to this call.
        std::vector<float> pcm(audio.size() / sizeof(float));
        std::memcpy(pcm.data(), audio.data(), audio.size());
        mono = resample_audio(pcm.data(), pcm.size() / channels, rate, channels,
                              engine->max_reference_samples(), format.sample_rate);
    }
    return ffi::ObjectRef(
        ffi::make_object<ReferenceHandle>(with_errors([&] {
            return engine->prepare_reference(mono, std::move(text), language);
        })));
}

std::vector<Input> requests_from(const ffi::Array<Values> &input) {
    std::vector<Input> requests;
    requests.reserve(input.size());
    for (const auto &r : input) {
        auto ref = r.at("reference");
        requests.push_back(
            {r.at("text").cast<std::string>(), r.at("instruction").cast<std::string>(),
             r.at("speaker").cast<std::string>(),
             ref == nullptr ? nullptr : ref.cast<const ReferenceHandle *>()->reference,
             r.at("language").cast<std::string>(),
             r.at("text_split_method").cast<std::string>(),
             r.at("fragment_interval").cast<double>()});
    }
    return requests;
}
GenerationOptions snapshot_options(const Options *options) {
    // Copy caller-owned options before enqueueing the request.
    GenerationOptions snapshot = *options;
    snapshot.synthesis_mode = parse_synthesis_mode(options->synthesis_mode_name);
    snapshot.seed = static_cast<uint64_t>(options->seed_bits);
    return snapshot;
}
void configure_logging(std::string level, std::string file, bool json) {
    log::configure({log::parse_level(level), file.empty() ? nullptr : file.c_str(), int(json)});
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(create, create);
void add_request(EngineHandle *handle, std::string request_id, ffi::Array<Values> input,
                 const Options *options) {
    with_errors([&] {
        handle->get()->add_request(
            {std::move(request_id), requests_from(input), snapshot_options(options)});
    });
}
Values pop_output(EngineHandle *handle, int timeout_ms) {
    auto frame = with_errors([&] {
        auto &queue = handle->get()->output_queue();
        std::unique_lock<std::mutex> lock(queue.mutex);
        if (timeout_ms < 0) queue.available.wait(lock, [&] { return queue.closed || !queue.frames.empty(); });
        else queue.available.wait_for(lock, std::chrono::milliseconds(timeout_ms),
            [&] { return queue.closed || !queue.frames.empty(); });
        if (queue.frames.empty()) {
            if (queue.closed) throw std::runtime_error("engine is shut down");
            return GenerationFrame{};
        }
        auto value = std::move(queue.frames.front()); queue.frames.pop_front();
        for (const auto &chunk : value.chunks) queue.bytes -= chunk.second.size() * sizeof(int16_t);
        return value;
    });
    if (frame.request_id.empty()) return {};
    std::string error;
    if (frame.error) { try { std::rethrow_exception(frame.error); }
        catch (const std::exception &failure) { error = failure.what(); } }
    ffi::Array<Values> chunks;
    for (const auto &[lane, pcm] : frame.chunks)
        chunks.push_back({{"index", lane},
                          {"pcm", ffi::Bytes(reinterpret_cast<const char *>(pcm.data()),
                                             pcm.size() * sizeof(int16_t))}});
    return {{"request_id", frame.request_id}, {"chunks", chunks}, {"error", error},
            {"batch_size", frame.batch_size}, {"done", frame.done},
            {"cancelled", frame.cancelled}, {"truncated", frame.stats.truncated}};
}
void cancel_request(EngineHandle *handle, std::string request_id) {
    handle->get()->cancel_request(request_id);
}
void configure_serving(EngineHandle *handle, int request_limit, int output_buffer_bytes) {
    if (request_limit < 1 || output_buffer_bytes < 1)
        throw std::invalid_argument("serving limits must be positive");
    handle->get()->configure_serving(request_limit, output_buffer_bytes);
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(configure_serving, configure_serving);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_request, add_request);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(pop_output, pop_output);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cancel_request, cancel_request);
void run_async(EngineHandle *handle) {
    handle->get()->run_async();
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_async, run_async);
void close(EngineHandle *handle) {
    handle->engine->shutdown();
}
std::string version() {
    return tts_version();
}
ffi::ObjectRef default_options(const EngineHandle *handle) {
    return ffi::ObjectRef(ffi::make_object<Options>(handle->get()->default_options()));
}
ffi::ObjectRef copy_options(const Options *options) {
    return ffi::ObjectRef(ffi::make_object<Options>(*options));
}
Values audio_format(const EngineHandle *handle) {
    auto f = handle->get()->audio_format();
    return {{"sample_rate", f.sample_rate},
            {"samples_per_frame", f.samples_per_frame},
            {"codebooks", f.codebooks},
            {"codebook_size", f.codebook_size}};
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(audio_format, audio_format);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(prepare_reference, reference);
ffi::Map<ffi::String, ffi::Any> model_info(EngineHandle *handle) {
    const auto &info = handle->get()->model_info();
    ffi::Array<ffi::String> files;
    for (const auto &name : info.checkpoint_files)
        files.push_back(name);
    return {{"model_type", info.model_type},
            {"frame_rate", double(info.audio.sample_rate) / info.audio.samples_per_frame},
            {"max_new_tokens", info.max_output_frames},
            {"checkpoint_files", files}};
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(model_info, model_info);
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
TVM_FFI_DLL_EXPORT_TYPED_FUNC(reference_from_codes, reference_from_codes);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(reference_codes, reference_codes);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(reference_frames, reference_frames);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(configure_logging, configure_logging);
} // namespace tts::python
