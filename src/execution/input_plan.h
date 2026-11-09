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
// Model-defined preparation transferred by Engine as owned records.
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
// CPU-owned model input, transferred once from Engine to Runner. Its model
// defines token layout, auxiliary features and validation; Engine passes it through unchanged.
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
    struct Region {
        std::string name;
        std::vector<int> lengths;
        int maximum_time;
    };
    enum class Replay { None, Batch, Cache, Regions };
    std::string name;
    InputBatch input;
    std::vector<std::string> outputs;
    ForwardMode mode = ForwardMode::Prefill;
    Replay replay = Replay::None;
    std::string history;
    std::vector<Region> regions;
};
// Construct and finish on Engine; transfer GPU input to Runner by ownership.
struct PreparedTask {
    std::unique_ptr<PreparationTask> task;
    ProgramSpec program;
    std::vector<PreparationOutput> outputs;
    std::exception_ptr error;
    HostResults results;
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
} // namespace tts
