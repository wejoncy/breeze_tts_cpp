#include "tts/engine.h"
#include "engine/request_scheduler.h"
#include "execution/model_runtime.h"
#include <chrono>
#include "support/log.h"
namespace tts {
constexpr double bytes_per_mib = 1024.0 * 1024.0;
struct Engine::Impl {
    const uint64_t id;
    std::unique_ptr<ModelRuntime> model;
    int max_batch;
    std::unique_ptr<RequestScheduler> scheduler;
    Impl(const std::string &dir, const ModelOptions &options, uint64_t engine_id)
        : id(engine_id), model(load_model(dir, options)), max_batch(options.max_batch) {
        scheduler = std::make_unique<RequestScheduler>(
            model->planner(), model->runner(), max_batch, model->scheduling_quantum_ms(), id,
            model->executor_initializer());
    }
};
void Engine::run_async() {
    log::check_not_callback();
    impl_->scheduler->run_async();
}
void Engine::add_request(const Request &request) {
    log::check_not_callback();
    impl_->scheduler->submit(request);
}
Engine::Frame Engine::read_result(const std::string &request_id, int timeout_ms) {
    log::check_not_callback();
    return impl_->scheduler->read(request_id, timeout_ms);
}
void Engine::cancel_request(const std::string &request_id) {
    log::check_not_callback();
    impl_->scheduler->cancel(request_id);
}
void Engine::shutdown() {
    log::check_not_callback();
    impl_->scheduler->stop();
}
void Engine::configure_serving(size_t request_limit, size_t output_buffer_bytes) {
    log::check_not_callback();
    impl_->scheduler->configure(request_limit, output_buffer_bytes);
}
Engine::Engine(const std::string &dir, int device, int batch, int context, int page,
               Quantization quantization, size_t runtime_memory_bytes, size_t memory_limit_bytes) {
    log::check_not_callback();
    auto id = log::next_engine_id();
    log::Scope scope(id);
    auto start = std::chrono::steady_clock::now();
    TTS_LOG(TTS_LOG_INFO, "engine",
            "loading device={} batch_capacity={} context={} quantization={}", device, batch,
            context,
            quantization == Quantization::None  ? "none"
            : quantization == Quantization::FP8 ? "fp8"
                                                : "int8");
    impl_ = std::make_unique<Impl>(dir, ModelOptions{device, batch, context, page, quantization,
                                                   runtime_memory_bytes, memory_limit_bytes}, id);
    if (log::enabled(TTS_LOG_INFO)) {
        auto memory = impl_->model->memory_stats();
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
    return impl_->model->info().weight_bytes;
}
MemoryStats Engine::memory_stats() const {
    log::check_not_callback();
    MemoryStats result;
    impl_->scheduler->run_sync([&] { result = impl_->model->memory_stats(); });
    return result;
}
AudioFormat Engine::audio_format() const {
    return impl_->model->info().audio;
}
const ModelInfo &Engine::model_info() const {
    return impl_->model->info();
}
GenerationOptions Engine::default_options() const {
    return impl_->model->default_options();
}
int Engine::batch_capacity() const {
    return impl_->max_batch;
}
size_t Engine::max_reference_samples() const {
    return impl_->model->info().max_reference_samples;
}
int Engine::generate(const std::vector<Input> &requests, const GenerationOptions &options,
                     AudioCallback callback, Stats &stats) {
    log::check_not_callback();
    auto id = impl_->scheduler->submit_generated(requests, options);
    try {
        while (true) {
            auto frame = read_result(id);
            stats = frame.stats;
            for (const auto &chunk : frame.chunks)
                if (!callback(chunk.first, chunk.second.data(), chunk.second.size())) {
                    cancel_request(id);
                    return 1;
                }
            if (frame.done)
                return frame.cancelled ? 1 : 0;
        }
    } catch (...) {
        cancel_request(id);
        throw;
    }
}

std::shared_ptr<const Reference> Engine::prepare_reference(const std::vector<float> &mono,
                                                         std::string transcript,
                                                         const std::string &language) {
    log::check_not_callback();
    std::shared_ptr<const Reference> reference;
    std::unique_ptr<ReferencePlan> plan;
    impl_->scheduler->run_sync(
        [&] { reference = impl_->model->runner().prepare_reference(*plan); },
        [&](const PreparationExecutor &execute) {
            plan = impl_->model->planner().prepare_reference(mono, std::move(transcript), execute,
                                                             language);
        });
    return reference;
}
std::vector<int> Engine::encode_audio(const std::vector<float> &mono) {
    log::check_not_callback();
    std::vector<int> codes;
    impl_->scheduler->run_sync([&] {
        if (impl_->model->info().audio.codebooks <= 0)
            throw std::invalid_argument("loaded model does not expose discrete audio encoding");
        codes = impl_->model->runner().encode_audio(mono);
        TTS_LOG(TTS_LOG_INFO, "reference", "encoded samples={} code_elements={}", mono.size(),
                codes.size());
    });
    return codes;
}
std::vector<float> Engine::decode_codes(const std::vector<int> &codes) {
    log::check_not_callback();
    std::vector<float> pcm;
    impl_->scheduler->run_sync([&] {
        if (impl_->model->info().audio.codebooks <= 0)
            throw std::invalid_argument("loaded model does not expose discrete audio decoding");
        pcm = impl_->model->runner().decode_codes(codes);
    });
    return pcm;
}
} // namespace tts
