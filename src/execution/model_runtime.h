#pragma once
#include "inputs/token_inputs.h"
#if TTS_ENABLE_GPT_SOVITS
#include "inputs/phoneme_inputs.h"
#endif
#include <variant>
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
// A finite set of concrete CPU frontends; no orchestration interface or callbacks.
using ModelInputs = std::variant<std::unique_ptr<TokenInputs>
#if TTS_ENABLE_GPT_SOVITS
    , std::unique_ptr<PhonemeInputs>
#endif
>;
struct ModelComponents {
    ModelInfo info;
    std::unique_ptr<Weights> parameters;
    std::unique_ptr<Model> network;
    ModelInputs inputs;
    std::unique_ptr<RunnerResources> resources;
    std::unique_ptr<ModelRunner> runner;
    GenerationOptions defaults;
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
    ModelInputs take_inputs();
    PagedKVCache &kv_cache();
    RequestSlots &cache_slots();
    ModelRunner &runner();
    DeviceContext &memory();
    MemoryStats memory_stats() const;
    void initialize_execution();
    void drain();
};
std::unique_ptr<ModelRuntime> load_model(const std::string &directory, const ModelOptions &);
} // namespace tts
