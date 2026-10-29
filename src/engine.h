#pragma once
#include "request.h"
#include "memory_stats.h"
#include "quantization.h"
#include <memory>
namespace tts {
class Engine {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    Engine(const std::string &, int device, int max_batch, int context, int kv_page_tokens = 128,
           Quantization quantization = Quantization::None);
    ~Engine();
    Engine(const Engine &) = delete;
    Engine &operator=(const Engine &) = delete;
    size_t weight_bytes() const;
    MemoryStats memory_stats() const;
    int batch_capacity() const;
    size_t max_reference_samples() const;
    int generate(const std::vector<Request> &, const GenerationOptions &, AudioCallback, Stats &);
    std::vector<int> encode_audio(const std::vector<float> &mono);
    std::vector<float> decode_codes(const std::vector<int> &codes);
};
} // namespace tts
