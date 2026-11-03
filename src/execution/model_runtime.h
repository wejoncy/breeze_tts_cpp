#pragma once
#include "execution/model_runner.h"
#include "tts/memory_stats.h"
#include "tts/quantization.h"
#include "tts/model_info.h"

namespace tts {
struct ModelOptions {
    int device, max_batch, context, kv_page_tokens;
    Quantization quantization;
    size_t runtime_memory_bytes, memory_limit_bytes;
};
// One loaded model, with distinct CPU planner and executor-owned GPU runner.
// The concrete model owns resource destruction order and device synchronization.
class ModelRuntime {
  public:
    virtual ~ModelRuntime() = default;
    virtual const ModelInfo &info() const = 0;
    virtual GenerationOptions default_options() const = 0;
    virtual RequestPlanner &planner() = 0;
    virtual ModelRunner &runner() = 0;
    virtual AudioCodec *audio_codec() {
        return nullptr;
    }
    virtual MemoryStats memory_stats() const = 0;
    // Acoustic models may execute several steps before producing a chunk.
    virtual double scheduling_quantum_ms() const = 0;
};
std::unique_ptr<ModelRuntime> load_model(const std::string &directory, const ModelOptions &);
} // namespace tts
