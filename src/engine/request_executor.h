#pragma once
#include "execution/request_planner.h"
#include <functional>

namespace tts {
class ModelRunner;
// Plans move to the executor. Batch rows refer to executor-owned requests by ID.
struct ExecutionRequest {
    uint64_t id = 0;
    std::unique_ptr<RequestPlan> plan;
};
struct ExecutionRow {
    uint64_t id;
    uint64_t request;
    size_t input;
};
struct ExecutionBatch {
    std::vector<ExecutionRequest> requests;
    std::vector<ExecutionRow> rows;
    std::vector<uint64_t> release;
    std::vector<uint64_t> release_requests;
    std::vector<uint64_t> prefill;
    double prefill_budget_ms = 0;
};
struct CompletedRow {
    uint64_t id;
    Stats stats;
    std::vector<int16_t> pcm;
    std::exception_ptr error;
    bool done = false;
    bool prefilled = false;
};
struct Completion {
    std::vector<CompletedRow> rows;
    std::exception_ptr fatal;
    double decode_ms = 0;
    int decoded_rows = 0;
};
// One execution thread owns the runner, admissions and live GPU request states.
// The scheduling policy and client output queues are outside this boundary.
class RequestExecutor {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    RequestExecutor(ModelRunner &, uint64_t engine_id, std::function<void(Completion)>);
    ~RequestExecutor();
    void start();
    void submit(ExecutionBatch);
    void control(std::function<void(std::exception_ptr)>);
    void stop();
};
} // namespace tts
