#pragma once
#include "execution/input_plan.h"
#include "execution/forward_output.h"
#include "runtime/scheduled_batch.h"
#include "runtime/inference_metrics.h"

namespace tts {
// Data contracts between scheduling policy and execution. Neither side calls
// the other; Engine transfers these records and owns their ordering.
struct ExecutionRequest {
    uint64_t id = 0;
    std::unique_ptr<RequestPlan> plan;
};
struct CompletedRow {
    uint64_t id;
    std::vector<int16_t> pcm;
    std::exception_ptr error;
    bool done = false, prefilled = false, truncated = false;
};
struct Completion {
    std::vector<CompletedRow> rows;
    IterationStats stats;
    size_t valid_rows = 0;
    std::exception_ptr fatal;
    int decoded_rows = 0;
    bool executed = false;
};
struct ExecutionBatch {
    std::vector<ExecutionRequest> requests;
    std::vector<ExecutionRow> rows;
    std::vector<uint64_t> release, release_requests;
    Completion output;
    bool empty() const {
        return rows.empty() && requests.empty() && release.empty() && release_requests.empty();
    }
};
struct PreparationBatch {
    std::vector<PreparedTask> tasks;
};
} // namespace tts
