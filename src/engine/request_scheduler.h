#pragma once
#include "tts/request.h"
#include <memory>
#include <functional>
namespace tts {
class ModelRunner;
class RequestPlanner;
// CPU scheduling/tokenization and GPU execution communicate through queues.
// Each actor exclusively owns its mutable state; callers only exchange messages.
class RequestScheduler {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    RequestScheduler(RequestPlanner &, ModelRunner &, int batch, double frame_period_ms,
                     uint64_t engine_id);
    ~RequestScheduler();
    void run_async();
    void configure(size_t request_limit, size_t output_buffer_bytes);
    void submit(const Request &);
    // Synchronous convenience APIs request an internal ID from the scheduler.
    std::string submit_generated(const std::vector<Input> &, const GenerationOptions &);
    GenerationFrame read(const std::string &request_id, int timeout_ms);
    void cancel(const std::string &request_id);
    void stop();
    void run_sync(std::function<void()>);
};
} // namespace tts
