#include "engine.h"
#include "model_runner.h"
#include "model.h"
#include "execution_resources.h"
#include <mutex>
#include <chrono>
#include "log.h"
namespace tts {
constexpr double bytes_per_mib = 1024.0 * 1024.0;
struct Engine::Impl {
    const uint64_t id;
    uint64_t request_sequence = 0;
    Backend gpu;
    const Model model;
    ExecutionResources resources;
    ModelRunner runner;
    int max_batch;
    std::mutex mutex;
    Impl(const std::string &dir, int device, int batch, int context, int page, ModelConfig config,
         uint64_t engine_id)
        : id(engine_id), gpu(device, config.backbone.quantization),
          model(gpu, dir, std::move(config)),
          resources(gpu.memory, model.backbone.config.layers,
                    model.backbone.config.kvheads * model.backbone.config.dim,
                    model.backbone.config.type, batch * 3, context, page),
          runner(model, gpu, resources, batch, context), max_batch(batch) {
        gpu.reserve_attention_workspace(
            size_t(batch) * 3 * context *
            std::max(model.backbone.config.heads, 2 * model.text.config.heads));
    }
    ~Impl() {
        cudaSetDevice(gpu.device);
        cudaStreamSynchronize(gpu.stream);
    }
};
Engine::Engine(const std::string &dir, int device, int batch, int context, int page,
               Quantization quantization) {
    log::check_not_callback();
    auto id = log::next_engine_id();
    log::Scope scope(id);
    auto start = std::chrono::steady_clock::now();
    TTS_LOG(TTS_LOG_INFO, "engine",
            "loading device={} batch_capacity={} context={} quantization={}", device, batch,
            context,
            quantization == Quantization::None  ? "bf16"
            : quantization == Quantization::FP8 ? "fp8"
                                                : "int8");
    impl_ = std::make_unique<Impl>(dir, device, batch, context, page,
                                   load_model_config(dir, batch, context, quantization), id);
    if (log::enabled(TTS_LOG_INFO)) {
        auto memory = impl_->gpu.memory.stats();
        auto ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count();
        TTS_LOG(TTS_LOG_INFO, "engine",
                "loaded ms={:.3f} weights={:.0f} MiB pool_used={:.0f} MiB pool_reserved={:.0f} MiB",
                ms, weight_bytes() / bytes_per_mib, memory.used_bytes / bytes_per_mib,
                memory.reserved_bytes / bytes_per_mib);
    }
}
Engine::~Engine() = default;
size_t Engine::weight_bytes() const {
    return impl_->model.weights.size();
}
MemoryStats Engine::memory_stats() const {
    log::check_not_callback();
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->gpu.memory.sync();
    return impl_->gpu.memory.stats();
}
int Engine::batch_capacity() const {
    return impl_->max_batch;
}
size_t Engine::max_reference_samples() const {
    return size_t(impl_->model.config.max_reference_frames) * impl_->model.config.samples_per_frame;
}
int Engine::generate(const std::vector<Request> &requests, const GenerationOptions &options,
                     AudioCallback callback, Stats &stats) {
    log::check_not_callback();
    std::lock_guard<std::mutex> lock(impl_->mutex);
    log::Scope scope(impl_->id, ++impl_->request_sequence);
    TTS_LOG(TTS_LOG_DEBUG, "request", "start batch={} max_frames={} graph={}", requests.size(),
            options.max_frames, options.use_cuda_graph);
    int status = impl_->runner.generate(requests, options, std::move(callback), stats);
    TTS_LOG(TTS_LOG_INFO, "request",
            "complete batch={} frames={} audio_seconds={:.3f} prefill_ms={:.3f} ttfa_ms={:.3f} "
            "total_ms={:.3f} cancelled={} truncated={}",
            requests.size(), stats.frames, stats.audio_seconds, stats.prefill_ms,
            stats.first_audio_ms, stats.total_ms, status == 1, stats.truncated);
    if (log::enabled(TTS_LOG_DEBUG)) {
        auto memory = impl_->gpu.memory.stats();
        TTS_LOG(TTS_LOG_DEBUG, "memory",
                "pool_used={:.0f} MiB pool_reserved={:.0f} MiB pool_peak_reserved={:.0f} MiB",
                memory.used_bytes / bytes_per_mib, memory.reserved_bytes / bytes_per_mib,
                memory.peak_reserved_bytes / bytes_per_mib);
    }
    return status;
}
std::vector<int> Engine::encode_audio(const std::vector<float> &mono) {
    log::check_not_callback();
    std::lock_guard<std::mutex> lock(impl_->mutex);
    log::Scope scope(impl_->id, ++impl_->request_sequence);
    TTS_LOG(TTS_LOG_DEBUG, "reference", "encode samples={}", mono.size());
    auto codes = impl_->runner.encode_audio(mono);
    TTS_LOG(TTS_LOG_INFO, "reference", "encoded samples={} code_elements={}", mono.size(),
            codes.size());
    return codes;
}
std::vector<float> Engine::decode_codes(const std::vector<int> &codes) {
    log::check_not_callback();
    std::lock_guard<std::mutex> lock(impl_->mutex);
    log::Scope scope(impl_->id, ++impl_->request_sequence);
    TTS_LOG(TTS_LOG_DEBUG, "codec", "decode code_elements={}", codes.size());
    return impl_->runner.decode_codes(codes);
}
} // namespace tts
