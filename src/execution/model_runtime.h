#pragma once
#include "execution/model_runner.h"
#include "tts/memory_stats.h"
#include "tts/quantization.h"
#include "tts/model_info.h"
#include "core/model.h"

namespace tts {
struct ModelOptions {
    int device, max_batch, context, kv_page_tokens;
    Quantization quantization;
    size_t runtime_memory_bytes, memory_limit_bytes;
};
class DeviceContext;
class ModelLoader;
class Weights;
struct RunnerResources;
struct ModelComponents {
    ModelInfo info;
    std::unique_ptr<Weights> parameters;
    std::unique_ptr<Model> network;
    std::unique_ptr<RequestPlanner> planner;
    std::unique_ptr<RunnerResources> resources;
    std::unique_ptr<ModelRunner> runner;
    GenerationOptions defaults;
    double quantum_ms = 0;
};
using ModelFactory = ModelComponents (*)(DeviceContext &, const ModelLoader &, const ModelOptions &);

// Device and execution lifetime are common to every registered model.
class ModelRuntime {
    struct Impl;
    std::unique_ptr<Impl> impl_;
  public:
    ModelRuntime(const ModelLoader &, const ModelOptions &, ModelFactory);
    ~ModelRuntime();
    const ModelInfo &info() const;
    GenerationOptions default_options() const;
    RequestPlanner &planner();
    ModelRunner &runner();
    MemoryStats memory_stats() const;
    double scheduling_quantum_ms() const;
    std::function<void()> executor_initializer();
};
std::unique_ptr<ModelRuntime> load_model(const std::string &directory, const ModelOptions &);
} // namespace tts
