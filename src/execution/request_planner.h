#pragma once
#include "tts/request.h"
#include "core/input_batch.h"
#include "core/tensor_storage.h"
#include "core/synthesis_windows.h"
#include "core/sequence_rate.h"
#include <exception>
#include <map>

namespace tts {
using HostResults = std::map<std::string, Tensor, std::less<>>;
struct PreparationOutput {
    std::string name;
    int rows;
};
// Model-defined preparation exchanged by ownership through the executor queue.
// CPU tokenization and GPU conditioning can alternate without sharing state.
struct PreparationTask {
    virtual ~PreparationTask() = default;
    virtual std::string_view program() const = 0;
    virtual InputBatch input() const = 0;
    // Each output concatenates these row counts when compatible tasks merge.
    virtual std::vector<PreparationOutput> outputs() const = 0;
    virtual void accept(HostResults) = 0;
};
struct GenerationSegment {
    InputBatch input, audio;
    std::vector<NamedTensor> initializers;
    std::vector<int> lengths, capacities;
    int max_tokens = 0, position_offset = 0;
    uint64_t seed = 0;
};
struct GenerationInput {
    std::vector<GenerationSegment> segments;
    SynthesisWindows windows;
    SequenceRate rate{1, 1};
    int pause_frames = 0;
};
using PreparationExecutor =
    std::function<std::unique_ptr<PreparationTask>(std::unique_ptr<PreparationTask>)>;
// CPU-owned model input, transferred once from scheduler to executor. Its model
// defines token layout, auxiliary features and validation; the engine never reads it.
struct RequestPlan {
    GenerationOptions options;
    std::vector<GenerationInput> inputs;
    virtual ~RequestPlan() = default;
};
struct ReferencePlan {
    std::string transcript;
    std::string program;
    InputBatch input;
    std::vector<std::string> outputs;
    virtual ~ReferencePlan() = default;
};
struct ProgramSpec {
    enum class Replay { None, Batch, Cache, Regions };
    std::string name;
    InputBatch input;
    std::vector<std::string> outputs;
    ForwardMode mode = ForwardMode::Prefill;
    Replay replay = Replay::None;
    std::string history;
};
// Construct on the scheduler; move the complete CPU input through the queue.
struct PreparedTask {
    std::unique_ptr<PreparationTask> task;
    ProgramSpec program;
    std::vector<PreparationOutput> outputs;
    std::exception_ptr error;
    explicit PreparedTask(std::unique_ptr<PreparationTask> value) : task(std::move(value)) {
        try {
            if (!task) throw std::invalid_argument("empty preparation task");
            program.name = task->program();
            program.input = task->input();
            outputs = task->outputs();
            if (program.name.empty() || outputs.empty())
                throw std::invalid_argument("preparation task has no forward program or outputs");
            for (const auto &field : outputs) {
                if (field.name.empty() || field.rows < 0 ||
                    std::find(program.outputs.begin(), program.outputs.end(), field.name) != program.outputs.end())
                    throw std::invalid_argument("invalid preparation output partition");
                program.outputs.push_back(field.name);
            }
        } catch (...) { error = std::current_exception(); }
    }
};
struct InputDefinition {
    std::string main_program = "tokens", audio_program = "waveform", reference_program = "reference";
    Type type = Type::BF16;
    int codebooks = 1, vocabulary = 0, eos = 0, min_tokens = 0;
    int max_branches = 1, max_audio_frames = 1;
    int query_heads = 0, head_dimensions = 0, max_decode_frames = 0;
    bool eos_on_argmax = false, nucleus_first = false, include_threshold = true;
    std::vector<ProgramSpec> programs;
};
// CPU-owned continuation. GPU tasks cross the executor queue by ownership.
class RequestPreparation {
  public:
    virtual ~RequestPreparation() = default;
    virtual std::unique_ptr<PreparationTask> next() = 0;
    virtual void accept(std::unique_ptr<PreparationTask>) {
        throw std::logic_error("unexpected GPU preparation result");
    }
    virtual std::unique_ptr<RequestPlan> finish() = 0;
};
class ImmediatePreparation final : public RequestPreparation {
    std::unique_ptr<RequestPlan> plan_;
  public:
    explicit ImmediatePreparation(std::unique_ptr<RequestPlan> plan) : plan_(std::move(plan)) {}
    std::unique_ptr<PreparationTask> next() override { return {}; }
    std::unique_ptr<RequestPlan> finish() override { return std::move(plan_); }
};
class RequestPlanner {
  public:
    virtual ~RequestPlanner() = default;
    virtual const InputDefinition &definition() const = 0;
    virtual std::unique_ptr<RequestPreparation> begin(const std::vector<Input> &,
                                                      const GenerationOptions &) const = 0;
    std::unique_ptr<RequestPlan> prepare(const std::vector<Input> &inputs,
                                        const GenerationOptions &options,
                                        const PreparationExecutor &execute = {}) const {
        auto preparation = begin(inputs, options);
        while (auto task = preparation->next()) {
            if (!execute) throw std::logic_error("model preparation requires the executor");
            preparation->accept(execute(std::move(task)));
        }
        return preparation->finish();
    }
    virtual std::unique_ptr<ReferencePlan> prepare_reference(const std::vector<float> &mono,
                                                           std::string transcript,
                                                           const PreparationExecutor & = {},
                                                           const std::string &language = {}) const = 0;
    virtual std::shared_ptr<const Reference> finish_reference(ReferencePlan &, HostResults) const = 0;
};
} // namespace tts
