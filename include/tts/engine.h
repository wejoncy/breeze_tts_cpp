#pragma once
#include "tts/request.h"
#include "tts/memory_stats.h"
#include "tts/quantization.h"
#include <memory>
#include <utility>
namespace tts {
class Engine {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    Engine(const std::string &, int device, int max_batch, int context, int kv_page_tokens = 128,
           Quantization quantization = Quantization::None, size_t runtime_memory_bytes = 0,
           size_t memory_limit_bytes = 0);
    ~Engine();
    Engine(const Engine &) = delete;
    Engine &operator=(const Engine &) = delete;
    // Construction loads resources only. Start the scheduler and executor explicitly.
    // The owner serializes lifecycle calls and stops ingress before shutdown.
    // Repeated starts while running are harmless; shutdown is terminal.
    void run_async();
    size_t weight_bytes() const;
    MemoryStats memory_stats() const;
    int batch_capacity() const;
    AudioFormat audio_format() const;
    size_t max_reference_samples() const;
    using Frame = GenerationFrame;
    // New limits apply to future submissions; admitted output buffers keep their bound.
    void configure_serving(size_t request_limit, size_t output_buffer_bytes);
    // Snapshot and enqueue without waiting for the scheduler. Full queues reject
    // immediately; validation errors arrive through read_result(). IDs must be unique.
    void add_request(const Request &);
    // Zero timeout polls; -1 waits. A terminal result/error consumes the request ID.
    // Duplicate submission errors preserve the original request and its ID.
    Frame read_result(const std::string &request_id, int timeout_ms = -1);
    // Post cancellation without waiting. The scheduler discards unread output,
    // wakes readers and signals execution; unknown/already consumed IDs are ignored.
    void cancel_request(const std::string &request_id);
    // Cancel outstanding work and join both threads; safe before start and on repeat.
    void shutdown();
    int generate(const std::vector<Input> &, const GenerationOptions &, AudioCallback, Stats &);
    std::vector<int> encode_audio(const std::vector<float> &mono);
    std::vector<float> decode_codes(const std::vector<int> &codes);
};
} // namespace tts
