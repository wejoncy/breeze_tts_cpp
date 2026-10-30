#pragma once
#include "execution/prepared_batch.h"

namespace tts {
// CPU data transferred from the scheduler to the executor. Validation and
// tokenization complete for every input before any input can execute.
struct RequestPlan {
    GenerationOptions options;
    std::vector<PreparedBatch> inputs;
};
class RequestPlanner {
    const Tokenizer &tokenizer_;
    const ModelConfig &config_;
    int max_batch_, context_;

  public:
    RequestPlanner(const Tokenizer &tokenizer, const ModelConfig &config, int batch, int context)
        : tokenizer_(tokenizer), config_(config), max_batch_(batch), context_(context) {}
    std::unique_ptr<RequestPlan> prepare(const std::vector<Input> &,
                                         const GenerationOptions &) const;
};
} // namespace tts
