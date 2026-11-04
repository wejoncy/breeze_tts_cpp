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
    bool create = true;
};
struct CompletedRow {
    uint64_t id;
    Stats stats;
    std::vector<int16_t> pcm;
    std::exception_ptr error;
    bool done = false;
    bool prefilled = false;
    bool deferred = false;
};
struct Completion {
    std::vector<CompletedRow> rows;
    size_t valid_rows = 0;
    std::exception_ptr fatal;
    double decode_ms = 0;
    int decoded_rows = 0;
    bool executed = false;
    bool resources_released = false;
};
struct ExecutionBatch {
    std::vector<ExecutionRequest> requests;
    std::vector<ExecutionRow> rows;
    std::vector<uint64_t> release;
    std::vector<uint64_t> release_requests;
    double prefill_budget_ms = 0;
    Completion output;
};
// One execution thread owns the runner, admissions and live GPU request states.
// The scheduling policy and client output queues are outside this boundary.
class RequestExecutor {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    using PreparationCallback = std::function<void(size_t, PreparedTask)>;
    RequestExecutor(ModelRunner &, int capacity, uint64_t engine_id, std::function<void(Completion)>,
                     std::function<void()> initialize);
    ~RequestExecutor();
    void start();
    void submit(ExecutionBatch);
    void prepare_task(std::unique_ptr<PreparationTask>,
                      std::function<void(std::unique_ptr<PreparationTask>, std::exception_ptr)>);
    void prepare_tasks(std::vector<PreparedTask>, PreparationCallback);
    void control(std::function<void(std::exception_ptr)>);
    void stop();
};
} // namespace tts
