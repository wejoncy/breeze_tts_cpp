#pragma once
#include "tts/request.h"
#include "tts/memory_stats.h"
#include "tts/quantization.h"
#include "tts/model_info.h"
#include <memory>
#include <utility>
#include <deque>
#include <mutex>
#include <condition_variable>
namespace tts {
class Engine {
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string begin_generation(const std::vector<Input> &, const GenerationOptions &);

  public:
    Engine(const std::string &, int device, int max_batch, int context, int kv_page_tokens = 128,
           Quantization quantization = Quantization::None, size_t runtime_memory_bytes = 0,
           size_t memory_limit_bytes = 0);
    ~Engine();
    Engine(const Engine &) = delete;
    Engine &operator=(const Engine &) = delete;
    // Construction loads resources only. Start the engine execution loop explicitly.
    // The owner serializes lifecycle calls and stops ingress before shutdown.
    // Repeated starts while running are harmless; shutdown is terminal.
    void run_async();
    size_t weight_bytes() const;
    MemoryStats memory_stats() const;
    int batch_capacity() const;
    AudioFormat audio_format() const;
    const ModelInfo &model_info() const;
    GenerationOptions default_options() const;
    size_t max_reference_samples() const;
    using Frame = GenerationFrame;
    struct OutputQueue {
        std::mutex mutex;
        std::condition_variable available;
        std::deque<Frame> frames;
        size_t bytes = 0, limit = 1 << 20;
        bool closed = false;
    };
    // One FIFO for every request. Its consumer owns the HTTP connection routing.
    OutputQueue &output_queue();
    // Bound accepted work and total PCM buffered in the shared output queue.
    void configure_serving(size_t request_limit, size_t output_buffer_bytes);
    // Snapshot and enqueue without waiting for the scheduler. Full queues reject
    // immediately. Results and errors enter the shared output queue.
    // IDs must remain unique until the caller consumes their terminal output.
    // Repeated active IDs are ignored and logged.
    void add_request(const Request &);
    // Post cancellation without waiting. Already queued output stays in FIFO order.
    // Active work ends with a cancelled terminal frame; unknown IDs are ignored.
    void cancel_request(const std::string &request_id);
    // Cancel outstanding work and join the engine thread; safe before start and on repeat.
    void shutdown();
    // Compatibility overload. The sink is owned and invoked by the caller.
    int generate(const std::vector<Input> &, const GenerationOptions &, AudioCallback, Stats &);
    template <class AudioSink> int generate(const std::vector<Input> &inputs,
                                            const GenerationOptions &options,
                                            AudioSink &&sink, Stats &stats) {
        auto id = begin_generation(inputs, options);
        bool cancelled = false, terminal = false;
        std::exception_ptr sink_error;
        try {
            for (;;) {
                auto &queue = output_queue();
                std::unique_lock<std::mutex> lock(queue.mutex);
                queue.available.wait(lock, [&] { return queue.closed || !queue.frames.empty(); });
                if (queue.frames.empty()) throw std::runtime_error("engine is shut down");
                auto frame = std::move(queue.frames.front());
                queue.frames.pop_front();
                for (const auto &chunk : frame.chunks) queue.bytes -= chunk.second.size() * sizeof(int16_t);
                lock.unlock();
                if (frame.request_id != id)
                    throw std::logic_error("generate requires exclusive ownership of the output queue");
                terminal = frame.done;
                if (frame.error) std::rethrow_exception(frame.error);
                stats = frame.stats;
                if (!cancelled) {
                    for (const auto &chunk : frame.chunks) {
                        bool keep = false;
                        try { keep = sink(chunk.first, chunk.second.data(), chunk.second.size()); }
                        catch (...) { sink_error = std::current_exception(); }
                        if (!keep) {
                            cancelled = true;
                            if (!terminal) cancel_request(id);
                            break;
                        }
                    }
                }
                if (terminal) {
                    if (sink_error) std::rethrow_exception(sink_error);
                    return cancelled || frame.cancelled ? 1 : 0;
                }
            }
        } catch (...) {
            if (!terminal) cancel_request(id);
            throw;
        }
    }

    // Mono samples at audio_format().sample_rate. The loaded model prepares all
    // conditioning it needs; callers do not reconstruct a reference from codes.
    std::shared_ptr<const Reference> prepare_reference(const std::vector<float> &mono,
                                                     std::string transcript,
                                                     const std::string &language = {});
    std::vector<int> encode_audio(const std::vector<float> &mono);
    std::vector<float> decode_codes(const std::vector<int> &codes);
};
} // namespace tts
