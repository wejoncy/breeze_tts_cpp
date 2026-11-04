#include "execution/model_runtime.h"
#include "execution/runner_resources.h"
#include "runtime/cuda_context.h"
#include "io/model_loader.h"
#include "io/weights.h"
#include <cmath>

namespace tts {
struct ModelRuntime::Impl {
    CudaContext gpu;
    ModelComponents parts;
    Impl(const ModelLoader &loader, const ModelOptions &options, ModelFactory factory)
        : gpu(options.device, options.quantization, options.memory_limit_bytes) {
        ExecutionScope initialization_scope(gpu);
        parts = factory(gpu.memory, loader, options);
        const auto &info = parts.info;
        if (!parts.network || !parts.planner || !parts.resources ||
            info.model_type != loader.config().at("model_type").get<std::string>() ||
            info.audio.sample_rate < 1 || info.audio.samples_per_frame < 1 ||
            info.max_output_frames < 1 || !std::isfinite(parts.quantum_ms) || parts.quantum_ms <= 0)
            throw std::invalid_argument("invalid model execution description");
        parts.runner = std::make_unique<ModelRunner>(*parts.network, *parts.resources,
            *parts.planner, parts.info, options.runtime_memory_bytes);
    }
    ~Impl() {
        cudaSetDevice(gpu.device);
        cudaStreamSynchronize(gpu.stream);
    }
};
ModelRuntime::ModelRuntime(const ModelLoader &loader, const ModelOptions &options,
                           ModelFactory factory)
    : impl_(std::make_unique<Impl>(loader, options, factory)) {}
ModelRuntime::~ModelRuntime() = default;
const ModelInfo &ModelRuntime::info() const { return impl_->parts.info; }
GenerationOptions ModelRuntime::default_options() const { return impl_->parts.defaults; }
RequestPlanner &ModelRuntime::planner() { return *impl_->parts.planner; }
ModelRunner &ModelRuntime::runner() { return *impl_->parts.runner; }
double ModelRuntime::scheduling_quantum_ms() const { return impl_->parts.quantum_ms; }
std::function<void()> ModelRuntime::executor_initializer() {
    return [this] { initialize_execution(impl_->gpu); };
}
MemoryStats ModelRuntime::memory_stats() const {
    impl_->gpu.memory.sync();
    return impl_->gpu.memory.stats();
}
} // namespace tts
