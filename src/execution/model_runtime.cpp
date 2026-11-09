#include "execution/model_runtime.h"
#include "execution/runner_resources.h"
#include "runtime/cuda_context.h"
#include "io/model_loader.h"
#include "io/weights.h"
#include "support/init_log.h"

namespace tts {
struct ModelRuntime::Impl {
    CudaContext gpu;
    ModelComponents parts;
    Impl(const ModelLoader &loader, const ModelOptions &options, ModelFactory factory)
        : gpu(options.device, options.quantization, options.memory_limit_bytes) {
        ExecutionScope initialization_scope(gpu);
        log::InitPhase phase("model construction and text frontend");
        parts = factory(gpu.memory, loader, options);
        const auto &info = parts.info;
        if (!parts.network || !std::visit([](const auto &inputs) { return bool(inputs); }, parts.inputs) || !parts.resources ||
            info.model_type != loader.config().at("model_type").get<std::string>() ||
            info.audio.sample_rate < 1 || info.audio.samples_per_frame < 1 ||
            info.max_output_frames < 1)
            throw std::invalid_argument("invalid model execution description");
        phase.next("execution runner");
        parts.runner = std::make_unique<ModelRunner>(*parts.network, *parts.resources,
            std::visit([](const auto &inputs) -> const InputDefinition & { return inputs->definition(); }, parts.inputs), parts.info, options.runtime_memory_bytes);
    }
    ~Impl() {
        cudaSetDevice(gpu.device);
        cudaStreamSynchronize(gpu.stream);
        cudaStreamSynchronize(gpu.memory.copy_stream);
    }
};
ModelRuntime::ModelRuntime(const ModelLoader &loader, const ModelOptions &options,
                           ModelFactory factory)
    : impl_(std::make_unique<Impl>(loader, options, factory)) {}
ModelRuntime::~ModelRuntime() = default;
const ModelInfo &ModelRuntime::info() const { return impl_->parts.info; }
GenerationOptions ModelRuntime::default_options() const { return impl_->parts.defaults; }
ModelInputs ModelRuntime::take_inputs() { return std::move(impl_->parts.inputs); }
PagedKVCache &ModelRuntime::kv_cache() { return impl_->parts.resources->kv; }
RequestSlots &ModelRuntime::cache_slots() { return impl_->parts.resources->slots; }
ModelRunner &ModelRuntime::runner() { return *impl_->parts.runner; }
DeviceContext &ModelRuntime::memory() { return impl_->gpu.memory; }
void ModelRuntime::initialize_execution() { tts::initialize_execution(impl_->gpu); }
void ModelRuntime::drain() { impl_->gpu.memory.sync(); }
MemoryStats ModelRuntime::memory_stats() const {
    // Engine has observed a stream event before requesting this snapshot.
    return impl_->gpu.memory.stats();
}
} // namespace tts
