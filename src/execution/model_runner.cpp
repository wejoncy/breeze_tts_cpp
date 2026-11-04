#include "execution/model_runner.h"
#include "execution/input_staging.h"
#include "ops/basic.h"
#include "ops/audio.h"
#include "ops/sampling.h"
#include "ops/attention.h"
#include "core/input_fields.h"
#include "support/profiler.h"
#include <array>
#include <cmath>
#include <numeric>
#include <type_traits>

namespace tts {
namespace {
using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
std::vector<NamedTensor> output_views(const std::vector<std::string> &names) {
    std::vector<NamedTensor> result;
    for (const auto &name : names) result.push_back({name, {}});
    return result;
}
std::vector<NamedTensor> output_views(const std::vector<std::string> &&) = delete;
}
struct ModelRunner::Impl {
    const Model &network;
    RunnerResources &runtime;
    RequestPlanner &planner;
    const ModelInfo &info;
    const InputDefinition &definition;
    std::vector<ProgramSpec> programs;
    std::vector<Buffer> input_storage;
    std::unique_ptr<FlashDecodePlan> decode_plan;
    const int capacity, lane_capacity;
    Buffer samples, previous_codes, code_rows, owners, caches, audio_caches, histories;
    Buffer done, counts, positions, pcm;
    TensorView audio_output;
    int audio_stride = 0;
    size_t input_workspace = 0;
    std::vector<int> host_done;
    std::vector<int> host_layout;
    std::vector<int16_t> host_audio;
    struct State final : RequestState {
        Impl &owner;
        GenerationInput input;
        size_t segment = 0;
        int generated = 0, rendered = 0, audio_rendered = 0, windows = 0, segment_start_frames = 0;
        bool semantic_done = false;
        WindowSchedule schedule;
        std::vector<int16_t> pending;
        State(Impl &implementation, std::unique_ptr<RequestAdmission::RowLease> credit,
              std::unique_ptr<RequestSlots::Lease> slot, GenerationInput value, GenerationOptions options)
            : RequestState(implementation.runtime, std::move(credit), std::move(slot), std::move(options)),
              owner(implementation), input(std::move(value)) {}
        GenerationSegment &current() { return input.segments.at(segment); }
        const GenerationSegment &current() const { return input.segments.at(segment); }
    };
    int available(const State &state) const {
        return std::min(state.options.max_frames - state.segment_start_frames,
            state.input.rate.frames(state.generated, state.options.speed_factor));
    }
    int ready(const State &state) const {
        const double now = std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
        const auto &windows = state.input.windows;
        const int accepted = windows.ready(available(state), state.rendered, state.semantic_done, state.schedule, now);
        if (accepted && !state.semantic_done) {
            const double required = std::ceil((double(state.rendered) + accepted +
                windows.lookahead(state.rendered) + windows.holdback) * state.options.speed_factor /
                state.input.rate.frames(1));
            if (state.generated < required) return 0;
        }
        return accepted;
    }
    bool audio_pending(const State &state) const {
        return state.input.windows.mode == SynthesisMode::Offline && state.semantic_done &&
            state.rendered == available(state) && state.audio_rendered < state.rendered;
    }
    void emit(State &state, const AudioCallback &callback, const int16_t *audio, size_t count) const {
        if (!count || state.finished) return;
        const int frame_samples = info.audio.samples_per_frame;
        if (count % frame_samples || count / frame_samples > size_t(state.options.max_frames - state.metrics.frames))
            throw std::logic_error("audio delivery exceeds the admitted frame limit");
        if (!state.metrics.frames) state.metrics.first_audio_ms = milliseconds(state.start);
        if (!callback(0, audio, count)) state.finished = true;
        state.metrics.frames += int(count / frame_samples);
        state.metrics.audio_seconds += double(count) / info.audio.sample_rate;
    }
    ProgramSpec audio_description(const std::vector<State *> &rows) const {
        auto result = program(definition.audio_program, ForwardMode::Prefill);
        auto &input = result.input;
        if (!rows.front()->current().audio.bindings.empty() || !rows.front()->current().audio.uploads.empty()) {
            std::vector<const InputBatch *> sources;
            for (const auto *row : rows) sources.push_back(&row->current().audio);
            auto auxiliary = merge_inputs({sources.data(), sources.size()});
            for (const auto &field : auxiliary.bindings) {
                auto found = std::find_if(input.bindings.begin(), input.bindings.end(),
                    [&](const auto &entry) { return entry.name == field.name; });
                if (found == input.bindings.end()) input.bindings.push_back(field);
                else *found = field;
            }
            input.uploads.insert(input.uploads.end(), auxiliary.uploads.begin(), auxiliary.uploads.end());
            input.owners.insert(input.owners.end(), auxiliary.owners.begin(), auxiliary.owners.end());
        }
        auto append = [&](const char *name, auto extract) {
            using Value = decltype(extract(*rows.front()));
            constexpr Type type = std::is_same_v<Value, double> ? Type::F64
                : std::is_same_v<Value, float> ? Type::F32
                : sizeof(Value) == 8 ? Type::I64 : Type::I32;
            auto tensor = own_tensor(input, {int(rows.size()), 1}, type);
            auto *values = tensor.ptr<Value>();
            for (size_t row = 0; row < rows.size(); ++row) values[row] = extract(*rows[row]);
            input.bindings.push_back({name, tensor});
        };
        append("generated", [](const State &state) { return state.generated; });
        append("rendered", [](const State &state) { return state.rendered; });
        append("audio_rendered", [](const State &state) { return state.audio_rendered; });
        append("available_frames", [&](const State &state) { return available(state); });
        append("frames", [&](const State &state) { return ready(state); });
        append("history_frames", [](const State &state) { return std::min(state.rendered, state.input.windows.left); });
        append("future_frames", [&](const State &state) {
            return std::min(state.input.windows.lookahead(state.rendered), available(state) - state.rendered - ready(state));
        });
        append("seed", [](const State &state) { return state.options.seed; });
        append("windows", [](const State &state) { return uint64_t(state.windows); });
        return result;
    }
    struct StateBindings {
        std::vector<std::vector<TensorView>> values;
        std::vector<NamedValue<ArrayView<const TensorView>>> fields;
        void bind(ForwardContext &context) const { context.state = {fields.data(), fields.size()}; }
    };
    std::vector<State *> active_audio;
    StateBindings state_bindings(const std::vector<State *> &rows, int padded) const {
        StateBindings result;
        const auto &prototype = runtime.states.rows().front()->views();
        result.values.reserve(prototype.size());
        result.fields.reserve(prototype.size());
        for (const auto &field : prototype) {
            result.values.emplace_back();
            auto &values = result.values.back();
            values.reserve(padded);
            for (int row = 0; row < padded; ++row) {
                const auto &source = rows.empty() ? *runtime.states.rows().at(std::min<size_t>(row, capacity - 1))
                    : rows.at(std::min<size_t>(row, rows.size() - 1))->data;
                values.push_back(source.views().at(field.first));
            }
            result.fields.push_back({field.first, {values.data(), values.size()}});
        }
        return result;
    }
    const ProgramSpec &program(std::string_view name, ForwardMode mode) const {
        for (const auto &candidate : programs)
            if (candidate.name == name && candidate.mode == mode) return candidate;
        throw std::invalid_argument("input definition has no matching forward program");
    }
    std::array<NamedTensor, 8> bindings(int rows, int lanes) const {
        return {{{"sampling", samples.view(rows, sizeof(SamplingRow), Type::I8)},
                 {"previous_codes", previous_codes.view(rows, 1, Type::I64)},
                 {"code_rows", code_rows.view(rows, 1, Type::I64)},
                 {"owner_rows", owners.view(lanes, 1, Type::I32)},
                 {"done", done.view(rows, 1, Type::I32)},
                 {"audio_positions", positions.view(rows, 1, Type::I32)},
                 {"histories", histories.view(1, int(histories.size()), Type::I8)},
                 {"destination", TensorView{}}}};
    }
    void prepare_forward(const ProgramSpec &description, ForwardContext::Mode mode) {
        ForwardContext context(description.input.limits.batch,
            description.input.limits.time, description.input.limits.rows);
        context.mode = mode;
        context.batch.forward_mode = description.mode;
        context.batch.capacity = runtime.context;
        context.batch.cache = runtime.kv.view(runtime.states.rows().front()->operator[]("table").ptr<int>());
        context.batch.cache.layout.sequences = caches.ptr<CacheSequence>();
        auto state = state_bindings({}, capacity);
        state.bind(context);
        auto arguments = bindings(capacity, lane_capacity);
        auto outputs = output_views(description.outputs);
        run_forward(network, description.name, description.input,
            {arguments.data(), arguments.size()}, {outputs.data(), outputs.size()}, context);
    }
    size_t workspace(const ProgramSpec &description, ForwardContext::Mode mode) {
        return Tensor::plan([&] { prepare_forward(description, mode); });
    }
    struct PackedProgram {
        ProgramSpec description;
        size_t count = 0, workspace_bytes = 0;
    };
    PackedProgram pack(std::string_view name, const std::vector<std::string> &outputs,
                       ArrayView<const InputBatch *> sources, size_t bytes_per_sequence = 0) {
        if (!sources.size) throw std::invalid_argument("cannot pack an empty forward batch");
        PackedProgram result;
        auto fits = [&](size_t count) {
            ProgramSpec candidate{std::string(name), merge_inputs({sources.data, count}), outputs};
            const size_t required = workspace(candidate, ForwardContext::Plan);
            const size_t extra = Tensor::aligned(size_t(candidate.input.sequence.shape.batch) * bytes_per_sequence);
            if (!runtime.memory.can_allocate({required + extra})) return false;
            result = {std::move(candidate), count, required};
            return true;
        };
        // Check the whole ready batch once, then retain FIFO prefix selection
        // under memory pressure without assuming workspace monotonicity.
        if (!fits(sources.size)) {
            size_t count = 1;
            while (count < sources.size && fits(count)) ++count;
            if (!result.count) throw RequestOverloaded("forward input exceeds initialized runtime workspace");
        }
        return result;
    }
    HostResults evaluate(const ProgramSpec &description, size_t workspace_bytes) {
        DeviceArena storage(runtime.memory, workspace_bytes);
        Tensor::Scope scope(storage);
        const auto &input = description.input;
        ForwardContext context(input.limits.batch, input.limits.time, input.limits.rows);
        auto outputs = output_views(description.outputs);
        run_forward(network, description.name, input, {}, {outputs.data(), outputs.size()},
                    context, &runtime.memory);
        HostResults result;
        for (const auto &field : outputs) {
            auto tensor = Tensor::empty({field.value.rows, field.value.cols}, field.value.type);
            runtime.memory.download(tensor.data_ptr(), field.value.data, field.value.size());
            result.emplace(field.name, std::move(tensor));
        }
        return result;
    }
    void invoke(const ProgramSpec &description, int rows, int lanes, int partitions = 1,
                ForwardContext::Mode mode = ForwardContext::Execute, bool replay = true) {
        auto lease = runtime.execution.lease_workspace();
        auto input = description.input;
        const int time = input.sequence.shape.time;
        const int source_rows = input.sequence.shape.rows;
        input.sequence = {{}, {lanes, time, lanes * time}};
        for (auto &field : input.bindings)
            if (source_rows && field.value.rows == source_rows)
                field.value = field.value.slice(0, lanes * time);
        ForwardContext context(input.limits.batch,
                                input.limits.time, input.limits.rows);
        context.mode = mode;
        std::vector<std::vector<ForwardShape>> shapes;
        std::vector<NamedValue<ArrayView<const ForwardShape>>> examples;
        if (description.replay == ProgramSpec::Replay::Regions) {
            if (mode == ForwardContext::Capture) {
                shapes.reserve(description.regions.size());
                examples.reserve(description.regions.size());
                for (const auto &region : description.regions) {
                    auto lengths = region.lengths;
                    lengths.erase(std::remove_if(lengths.begin(), lengths.end(),
                        [&](int length) { return length < 1 || length > region.maximum_time; }), lengths.end());
                    std::sort(lengths.begin(), lengths.end(), std::greater<int>());
                    lengths.erase(std::unique(lengths.begin(), lengths.end()), lengths.end());
                    auto &samples = shapes.emplace_back();
                    const auto &batches = runtime.shapes.rows();
                    for (auto batch = batches.rbegin(); batch != batches.rend(); ++batch)
                        for (int length : lengths) {
                            if (*batch > INT32_MAX / length)
                                throw std::invalid_argument("graph shape exceeds tensor dimensions");
                            samples.push_back({*batch, length, *batch * length});
                        }
                    examples.push_back({region.name, {samples.data(), samples.size()}});
                }
                context.examples = {examples.data(), examples.size()};
            }
            context.allow_replay = replay;
            context.executor = &runtime.execution.graphs();
            context.dispatch = [](void *executor, const ForwardCall &call, bool capture) {
                auto &graphs = *static_cast<CUDAGraphRunner *>(executor);
                return capture ? graphs.capture(call) : graphs.run(call);
            };
        }
        context.batch.forward_mode = description.mode;
        context.batch.capacity = runtime.context;
        context.batch.decode_partitions = partitions;
        context.batch.cache = runtime.kv.view();
        context.batch.cache.layout.sequences = description.history.empty()
            ? caches.ptr<CacheSequence>() : audio_caches.ptr<CacheSequence>();
        auto state = state_bindings(active_audio, rows);
        state.bind(context);
        auto arguments = bindings(rows, lanes);
        auto outputs = output_views(description.outputs);
        for (auto &field : outputs)
            if (field.name == "audio_layout") {
                std::fill(host_layout.begin(), host_layout.end(), 0);
                field.value = TensorView(host_layout.data(), capacity, 2, Type::I32);
            }
        run_forward(network, description.name, input, {arguments.data(), arguments.size()},
                    {outputs.data(), outputs.size()}, context);
        if (mode != ForwardContext::Execute) return;
        TensorInputs result(outputs.data(), outputs.size());
        if (description.name == definition.main_program && description.mode == ForwardMode::Decode) {
            ops::advance_positions(done.view(rows, 1, Type::I32), nullptr, lanes, 1,
                                    caches.ptr<CacheSequence>());
        } else if (description.name == definition.audio_program) {
            auto selected = std::find_if(outputs.begin(), outputs.end(),
                [](const auto &field) { return field.name == "pcm" && field.value.numel(); });
            if (selected != outputs.end()) audio_output = selected->value;
            else {
                auto waveform = result["waveform"];
                if (waveform.numel()) tts::pcm(waveform, pcm.ptr<int16_t>());
                audio_output = pcm.view(waveform.rows, waveform.cols, Type::I16);
            }
            audio_stride = int(audio_output.numel() / rows);
        }
    }
    void map(const std::vector<State *> &rows, int padded_rows, int padded_lanes) {
        std::vector<SamplingRow> sampling(padded_rows);
        std::vector<const int *> previous(padded_rows), codes(padded_rows);
        std::vector<CacheSequence> sequences(padded_lanes);
        std::vector<int> membership(padded_lanes, -1), generated(padded_rows), relative(padded_rows);
        int lane_index = 0;
        for (size_t row = 0; row < rows.size(); ++row) {
            auto &state = *rows[row];
            auto &data = state.data;
            const size_t token_bytes = size_t(definition.codebooks) * sizeof(int);
            const bool history = data["codes"].size() > token_bytes;
            auto *tokens = data["codes"].ptr<int>();
            previous[row] = tokens + size_t(history ? std::max(0, state.generated - 1) : 0) * definition.codebooks;
            codes[row] = tokens + size_t(history ? state.generated : 0) * definition.codebooks;
            auto &sample = sampling[row];
            sample.codes = const_cast<int *>(codes[row]);
            sample.done = data["done"].ptr<int>();
            sample.seen = data["seen"].ptr<int>();
            sample.rng = data["rng"].ptr<uint64_t>();
            sample.branches = int(state.current().lengths.size());
            sample.cfg = state.options.cfg_scale;
            sample.cfg_ref = state.options.cfg_scale_ref;
            sample.cfg_ins = state.options.cfg_scale_ins;
            sample.penalty = state.options.repetition_penalty;
            sample.backbone = {state.options.do_sample ? state.options.temperature : 0.f,
                state.options.top_p, state.options.top_k, definition.nucleus_first, definition.include_threshold};
            sample.depth = {state.options.depth_do_sample ? state.options.depth_temperature : 0.f,
                state.options.depth_top_p, state.options.depth_top_k};
            sample.generated = counts.ptr<int>() + row;
            sample.eos_min_tokens = definition.min_tokens;
            sample.eos_on_argmax = definition.eos_on_argmax;
            generated[row] = state.generated;
            relative[row] = state.current().position_offset + std::max(0, state.generated - 1);
            auto cache = state.cache->view();
            for (int branch = 0; branch < sample.branches; ++branch) {
                sample.logits[branch] = lane_index;
                membership[lane_index] = int(row);
                sequences[lane_index++] = {cache.keys.data, cache.values.data,
                    size_t(cache.rows_per_layer) * cache.keys.cols * bytes(cache.keys.type),
                    cache.layout.block_table + branch * cache.layout.block_stride,
                    data["positions"].ptr<int>() + branch};
            }
        }
        runtime.memory.upload(samples.data(), sampling.data(), sampling.size() * sizeof(SamplingRow));
        runtime.memory.upload(previous_codes.data(), previous.data(), previous.size() * sizeof(int *));
        runtime.memory.upload(code_rows.data(), codes.data(), codes.size() * sizeof(int *));
        runtime.memory.upload(owners.data(), membership.data(), membership.size() * sizeof(int));
        runtime.memory.upload(caches.data(), sequences.data(), sequences.size() * sizeof(CacheSequence));
        runtime.memory.upload(counts.data(), generated.data(), generated.size() * sizeof(int));
        runtime.memory.upload(positions.data(), relative.data(), relative.size() * sizeof(int));
    }
    void sampled(const std::vector<State *> &rows) {
        runtime.memory.download(host_done.data(), done.data(), rows.size() * sizeof(int));
        for (size_t row = 0; row < rows.size(); ++row) {
            auto &state = *rows[row];
            if (host_done[row]) state.semantic_done = true;
            else if (++state.generated == state.current().max_tokens) {
                state.semantic_done = true;
                state.metrics.truncated = 1;
            }
        }
    }
    void start_segment(State &state) {
        auto &input = state.current();
        state.data.reset_host();
        state.cache.reset();
        state.cache = std::make_unique<PagedKVCache::Lease>(runtime.kv, input.lengths, input.capacities,
                                                          state.data["table"].ptr<int>());
        for (auto name : {"codes", "done", "seen"}) runtime.memory.zero(state.data[name]);
        state.options.seed = input.seed;
        runtime.memory.upload(state.data["rng"].data(), &input.seed, sizeof(input.seed));
        runtime.memory.upload(state.data["positions"].data(), input.lengths.data(), input.lengths.size() * sizeof(int));
        for (const auto &field : input.initializers) {
            auto &destination = state.data[field.name];
            if (field.value.size() > destination.size()) throw std::invalid_argument("initial state exceeds its declared capacity");
            runtime.memory.upload(destination.data(), field.value.data, field.value.size());
        }
        for (const auto &address : input.input.addresses) {
            auto values = std::make_shared<Tensor>(Tensor::zeros({int(address.offsets.size()), 1}, Type::I64));
            auto *pointers = values->view().ptr<uint64_t>();
            auto &buffer = state.data[address.state];
            for (size_t row = 0; row < address.offsets.size(); ++row) {
                const auto offset = address.offsets[row];
                if (offset >= 0 && uint64_t(offset) >= buffer.size()) throw std::invalid_argument("state input address exceeds its field");
                pointers[row] = offset < 0 ? 0 : reinterpret_cast<uint64_t>(buffer.ptr<char>() + offset);
            }
            input.input.uploads.push_back({address.tensor, values->view()});
            input.input.owners.push_back(values);
        }
        for (const auto &description : programs)
            if (!description.history.empty()) state.data.history(description.history).reset(runtime.memory.stream);
        state.generated = state.rendered = state.audio_rendered = state.windows = 0;
        state.segment_start_frames = state.metrics.frames;
        const double now = std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
        state.schedule = WindowSchedule(state.input.windows.core, state.input.windows.maximum, now);
        state.pending.clear();
        state.pending.reserve(size_t(state.input.windows.fade) * info.audio.samples_per_frame);
        state.is_prefilled = state.semantic_done = false;
    }
    Impl(const Model &model, RunnerResources &resources, RequestPlanner &inputs,
          const ModelInfo &metadata, size_t runtime_bytes)
        : network(model), runtime(resources), planner(inputs), info(metadata), definition(inputs.definition()), programs(definition.programs),
          capacity(resources.batch), lane_capacity(capacity * definition.max_branches),
          samples(size_t(capacity) * sizeof(SamplingRow)),
          previous_codes(size_t(capacity) * sizeof(void *)),
          code_rows(previous_codes.size()), owners(size_t(lane_capacity) * sizeof(int)),
          caches(size_t(lane_capacity) * sizeof(CacheSequence)),
          audio_caches(size_t(capacity) * sizeof(CacheSequence)),
          histories([&] {
              size_t maximum = 0;
              for (const auto &description : programs)
                  if (!description.history.empty()) maximum = std::max(maximum,
                      resources.states.rows().front()->history(description.history).history.size());
              return maximum * capacity * sizeof(ConvHistory);
          }()), done(size_t(capacity) * sizeof(int)),
          counts(done.size()), positions(done.size()),
          pcm(size_t(capacity) * definition.max_audio_frames * info.audio.samples_per_frame * sizeof(int16_t)),
          host_done(capacity), host_layout(size_t(capacity) * 2), host_audio(pcm.size() / sizeof(int16_t)) {
        if (definition.query_heads > 0) {
            decode_plan = std::make_unique<FlashDecodePlan>(std::max(1,
                ops::decode_capacity(runtime.memory.device, definition.head_dimensions, definition.type)),
                runtime.context);
            ops::reserve_decode(runtime.memory.device, definition.head_dimensions, definition.type);
        }
        for (auto *buffer : {&samples, &previous_codes, &code_rows, &owners, &caches,
                            &audio_caches, &histories, &done, &counts, &positions}) runtime.memory.zero(*buffer);
        for (auto &description : programs) {
            if (description.replay == ProgramSpec::Replay::None) continue;
            for (const auto &field : description.input.uploads) {
                input_storage.emplace_back(field.value.size());
                auto &buffer = input_storage.back();
                runtime.memory.zero(buffer);
                description.input.bindings.push_back({field.name,
                    buffer.view(field.value.rows, field.value.cols, field.value.type)});
            }
            description.input.uploads.clear();
        }
        size_t retained = 0, preparation = 0;
        for (const auto &description : programs) {
            preparation = std::max(preparation, workspace(description, ForwardContext::Prepare));
            const size_t bytes = workspace(description, ForwardContext::Plan);
            if (description.replay != ProgramSpec::Replay::None || description.name == definition.audio_program)
                retained = std::max(retained, bytes);
            if (description.replay == ProgramSpec::Replay::None && description.name != definition.audio_program)
                input_workspace = std::max(input_workspace, bytes);
        }
        runtime.execution.prepare(retained);
        {
            DeviceArena storage(runtime.memory, preparation);
            for (const auto &description : programs) {
                Tensor::Scope scope(storage);
                prepare_forward(description, ForwardContext::Prepare);
            }
        }
        auto &graphs = runtime.execution.graphs();
        for (const auto &description : programs) {
            if (description.replay == ProgramSpec::Replay::Cache) {
                for (int rows : runtime.shapes.rows())
                    for (int branches = 1; branches <= definition.max_branches; ++branches)
                        for (int partitions : decode_plan
                                ? decode_plan->graph_splits(rows * branches, definition.query_heads)
                                : std::vector<int>{1})
                            graphs.register_shape(description.name.c_str(), {rows, rows * branches, partitions});
                graphs.capture(description.name.c_str(), [&](const auto &shape) {
                    invoke(description, shape[0], shape[1], shape[2]);
                });
            } else if (description.replay == ProgramSpec::Replay::Batch) {
                for (int rows : runtime.shapes.rows()) graphs.register_shape(description.name.c_str(), {rows});
                graphs.capture(description.name.c_str(), [&](const auto &shape) { invoke(description, shape[0], shape[0]); });
            } else if (description.replay == ProgramSpec::Replay::Regions) {
                invoke(description, capacity, capacity, 1, ForwardContext::Capture);
            }
        }
        runtime.execution.seal();
        ops::seal(runtime.memory.device);
        runtime.memory.trim();
        size_t transfers = pcm.size();
        for (const auto &description : definition.programs) {
            const size_t upload = Tensor::plan([&] {
                for (const auto &field : description.input.uploads) Tensor::reserve(field.value.size());
                Tensor::reserve(description.input.sequence.metadata.size());
                for (const auto &sequence : description.input.sequences) Tensor::reserve(sequence.value.metadata.size());
                for (const auto &region : description.input.regions)
                    if (region.tensor.device < 0) Tensor::reserve(region.tensor.size());
            });
            transfers = std::max(transfers, upload);
        }
        for (auto *buffer : {&samples, &previous_codes, &code_rows, &owners, &caches,
                            &audio_caches, &histories, &done, &counts, &positions})
            transfers += Tensor::aligned(buffer->size());
        for (const auto &row : runtime.states.rows())
            for (const auto &field : row->fields()) transfers += Tensor::aligned(field.second.size());
        runtime.memory.reserve_transfers(transfers);
        const size_t available = runtime.memory.available_bytes();
        const size_t budget = runtime_bytes ? runtime_bytes
            : std::min(input_workspace + runtime.kv.max_storage_bytes(), available - available / 10);
        if (!budget || budget > available) throw DeviceOutOfMemory("runtime reservation exceeds the available memory budget");
        runtime.memory.reserve_runtime(budget);
    }
    State &state(RequestState &value) {
        auto &result = dynamic_cast<State &>(value);
        if (&result.owner != this) throw std::invalid_argument("request belongs to another execution engine");
        return result;
    }
};
ModelRunner::ModelRunner(const Model &model, RunnerResources &resources, RequestPlanner &inputs,
                         const ModelInfo &info, size_t runtime_bytes)
    : impl_(std::make_unique<Impl>(model, resources, inputs, info, runtime_bytes)) {}
ModelRunner::~ModelRunner() = default;
size_t ModelRunner::max_step_samples() const {
    return size_t(impl_->definition.max_audio_frames) * impl_->info.audio.samples_per_frame +
           size_t(impl_->info.audio.sample_rate) * 10;
}
std::unique_ptr<ModelRunner::RequestAdmission> ModelRunner::admit(RequestPlan &plan) {
    auto &implementation = *impl_;
    std::vector<int> pages;
    size_t workspace = 0;
    for (const auto &input : plan.inputs) {
        int maximum = 0;
        for (const auto &segment : input.segments) {
            maximum = std::max(maximum, implementation.runtime.kv.required_pages(segment.capacities));
            ProgramSpec description{implementation.definition.main_program, segment.input, {"hidden_states", "logits", "done"}};
            workspace = std::max(workspace, implementation.workspace(description, ForwardContext::Plan));
        }
        pages.push_back(maximum);
    }
    return implementation.runtime.admissions.admit(pages, workspace);
}
std::unique_ptr<ModelRunner::RequestState> ModelRunner::prepare(RequestPlan &plan, size_t input,
                                                               RequestAdmission &admission) {
    auto &implementation = *impl_;
    if (!admission.belongs_to(implementation.runtime.admissions)) throw std::invalid_argument("foreign request admission");
    auto credit = admission.acquire_row();
    if (!credit) return {};
    auto state = std::make_unique<Impl::State>(implementation, std::move(credit),
        implementation.runtime.slots.acquire(), std::move(plan.inputs.at(input)), plan.options);
    implementation.start_segment(*state);
    return state;
}
size_t ModelRunner::prefill_step(const std::vector<RequestState *> &requests) {
    auto &implementation = *impl_;
    std::vector<Impl::State *> rows;
    std::vector<const InputBatch *> sources;
    for (auto *request : requests) {
        auto &state = implementation.state(*request);
        if (state.is_prefilled || state.finished) continue;
        rows.push_back(&state);
        sources.push_back(&state.current().input);
    }
    if (rows.empty()) return 0;
    auto program = implementation.pack(implementation.definition.main_program,
        {"hidden_states", "logits", "done"}, {sources.data(), sources.size()},
        size_t(implementation.runtime.kv.block_stride()) * sizeof(int));
    rows.resize(program.count);
    auto &packed = program.description.input;
    std::vector<int> blocks;
    blocks.reserve(size_t(packed.sequence.shape.batch) * implementation.runtime.kv.block_stride());
    for (const auto *state : rows) {
        const auto &table = state->cache->block_table();
        blocks.insert(blocks.end(), table.begin(), table.end());
    }
    const size_t bytes = program.workspace_bytes + Tensor::aligned(blocks.size() * sizeof(int));
    DeviceArena storage(implementation.runtime.memory, bytes);
    auto block_values = std::make_shared<Tensor>(Tensor::zeros({int(blocks.size()), 1}, Type::I32));
    std::copy(blocks.begin(), blocks.end(), block_values->view().ptr<int>());
    packed.uploads.push_back({"cache_blocks", block_values->view()});
    packed.owners.push_back(block_values);
    packed.batch_bindings.push_back({"cache_blocks", BatchField::CacheTable});
    implementation.map(rows, int(rows.size()), packed.sequence.shape.batch);
    Tensor::Scope scope(storage);
    ForwardContext context(packed.limits.batch, packed.limits.time, packed.limits.rows);
    context.batch.capacity = implementation.runtime.context;
    context.batch.cache = implementation.runtime.kv.view();
    auto arguments = implementation.bindings(int(rows.size()), packed.sequence.shape.batch);
    auto outputs = output_views(program.description.outputs);
    run_forward(implementation.network, program.description.name, packed, {arguments.data(), arguments.size()},
                {outputs.data(), outputs.size()}, context, &implementation.runtime.memory);
    implementation.sampled(rows);
    for (auto *state : rows) {
        state->is_prefilled = true;
        state->metrics.prefill_ms = milliseconds(state->start);
    }
    return rows.size();
}
void ModelRunner::step(const std::vector<RequestState *> &requests,
                        const std::vector<AudioCallback> &callbacks) {
    if (requests.empty() || requests.size() != callbacks.size()) throw std::invalid_argument("invalid execution batch");
    auto &implementation = *impl_;
    auto &runtime = implementation.runtime;
    std::vector<Impl::State *> rows, decode;
    implementation.active_audio.clear();
    auto available = [&](const Impl::State &state) { return implementation.available(state); };
    auto ready = [&](const Impl::State &state) { return implementation.ready(state); };
    for (auto *request : requests) {
        auto &state = implementation.state(*request);
        if (!state.is_prefilled) throw std::logic_error("decode scheduled before input execution");
        rows.push_back(&state);
        if (!state.finished && !state.semantic_done && !ready(state)) {
            auto lengths = state.current().lengths;
            for (auto &length : lengths) length += state.generated;
            state.cache->ensure(lengths);
            decode.push_back(&state);
        }
    }
    if (!decode.empty()) {
        int lanes = 0, key_length = 0;
        bool enabled = true;
        for (const auto *state : decode) {
            lanes += int(state->current().lengths.size());
            for (int length : state->current().lengths)
                key_length = std::max(key_length, length + state->generated);
            enabled &= state->options.use_cuda_graph;
        }
        auto shape = runtime.shapes.select(int(decode.size()), lanes, implementation.definition.max_branches);
        if (!shape.first) shape = {int(decode.size()), lanes};
        implementation.map(decode, shape.first, shape.second);
        const int partitions = implementation.decode_plan
            ? implementation.decode_plan->select(shape.second, implementation.definition.query_heads, key_length) : 1;
        const auto &description = implementation.program(implementation.definition.main_program, ForwardMode::Decode);
        runtime.execution.graphs().run(description.name.c_str(), {shape.first, shape.second, partitions}, enabled,
            [&] { implementation.invoke(description, shape.first, shape.second, partitions); });
        implementation.sampled(decode);
    }
    const auto &description = implementation.program(implementation.definition.audio_program, ForwardMode::Prefill);
    std::vector<Impl::State *> audio;
    std::vector<size_t> members;
    for (size_t row = 0; row < rows.size(); ++row)
        if (!rows[row]->finished && (ready(*rows[row]) || implementation.audio_pending(*rows[row]))) {
            members.push_back(row);
            audio.push_back(rows[row]);
        }
    if (!audio.empty()) {
        int padded = description.replay == ProgramSpec::Replay::None ? int(audio.size())
            : runtime.shapes.select(int(audio.size()));
        if (!padded) padded = int(audio.size());
        std::vector<const int *> codes(padded);
        std::vector<CacheSequence> sequences(padded);
        const size_t history_count = description.history.empty() ? 0
            : runtime.states.history_spec(description.history).history_bytes.size();
        std::vector<ConvHistory> history(history_count * implementation.capacity);
        bool enabled = true;
        for (size_t row = 0; row < audio.size(); ++row) {
            auto &state = *audio[row];
            const size_t books = implementation.definition.codebooks;
            const bool retained = state.data["codes"].size() > books * sizeof(int);
            codes[row] = state.data["codes"].ptr<int>() + size_t(retained ? state.generated - 1 : 0) * books;
            enabled &= state.options.use_cuda_graph;
            if (!description.history.empty()) {
                auto &sequence = state.data.history(description.history);
                sequences[row] = {sequence.keys.data(), sequence.values.data(), sequence.layer_bytes,
                                  nullptr, sequence.position.ptr<int>()};
                for (size_t layer = 0; layer < history_count; ++layer)
                    history[layer * implementation.capacity + row] = {
                        sequence.history[layer].ptr<float>(), sequence.position.ptr<int>()};
            }
        }
        runtime.memory.upload(implementation.code_rows.data(), codes.data(), codes.size() * sizeof(int *));
        runtime.memory.upload(implementation.audio_caches.data(), sequences.data(), sequences.size() * sizeof(CacheSequence));
        if (!history.empty()) runtime.memory.upload(implementation.histories.data(), history.data(), history.size() * sizeof(ConvHistory));
        implementation.active_audio = audio;
        auto execution = implementation.audio_description(audio);
        runtime.execution.graphs().run(description.name.c_str(), {padded}, enabled,
            [&] { implementation.invoke(execution, padded, padded, 1, ForwardContext::Execute, enabled); });
        const bool indexed = std::find(description.outputs.begin(), description.outputs.end(), "audio_layout") != description.outputs.end();
        const int stride = implementation.audio_stride;
        const size_t total = indexed ? implementation.audio_output.numel() : size_t(stride) * audio.size();
        if ((!indexed && stride < 1) || total > implementation.host_audio.size())
            throw std::logic_error("model audio exceeds its declared output capacity");
        if (total) runtime.memory.download(implementation.host_audio.data(), implementation.audio_output.data, total * sizeof(int16_t));
        const TensorInputs audio_inputs(execution.input.bindings.data(), execution.input.bindings.size());
        for (size_t row = 0; row < audio.size(); ++row) {
            auto &state = *audio[row];
            const int accepted = audio_inputs["frames"].ptr<int>()[row];
            const int left = std::min(state.rendered, state.input.windows.left);
            const int offset = indexed ? implementation.host_layout[row * 2] : int(row) * stride;
            const int count = indexed ? implementation.host_layout[row * 2 + 1] : stride;
            const int samples_per_frame = implementation.info.audio.samples_per_frame;
            const bool deferred = state.input.windows.mode == SynthesisMode::Offline;
            state.rendered += accepted;
            if (accepted) ++state.windows;
            if (!count) continue;
            const int begin = left * samples_per_frame, length = deferred ? count : accepted * samples_per_frame;
            if (offset < 0 || count < begin + length || size_t(offset + count) > total ||
                state.pending.size() > size_t(begin)) throw std::logic_error("invalid model audio output layout");
            auto *current = implementation.host_audio.data() + offset + begin;
            auto *output = current - state.pending.size();
            for (size_t sample = 0; sample < state.pending.size(); ++sample) {
                const float alpha = state.pending.size() > 1 ? float(sample) / (state.pending.size() - 1) : 1;
                output[sample] = int16_t(std::lround(state.pending[sample] * (1 - alpha) + output[sample] * alpha));
            }
            const bool final = state.semantic_done && state.rendered == available(state);
            const int send = final ? length : length - state.input.windows.fade * samples_per_frame;
            const size_t emitted = state.pending.size() + send;
            state.pending.assign(current + send, current + length);
            implementation.emit(state, callbacks[members[row]], output, emitted);
            state.audio_rendered += int(emitted / samples_per_frame);
            if (state.input.windows.mode == SynthesisMode::Hybrid) {
                const double now = std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
                state.schedule.observe(accepted, emitted, implementation.info.audio.sample_rate, now);
            }
        }
        implementation.active_audio.clear();
    }
    for (size_t row = 0; row < rows.size(); ++row) {
        auto *state = rows[row];
        if (!state->finished && state->semantic_done && state->rendered >= available(*state) &&
            !implementation.audio_pending(*state)) {
            implementation.emit(*state, callbacks[row], state->pending.data(), state->pending.size());
            state->pending.clear();
            if (state->segment + 1 < state->input.segments.size() && state->metrics.frames < state->options.max_frames) {
                int pause = std::min(state->input.pause_frames, state->options.max_frames - state->metrics.frames);
                std::fill(implementation.host_audio.begin(), implementation.host_audio.end(), int16_t{});
                while (pause && !state->finished) {
                    const int count = std::min(pause, implementation.definition.max_audio_frames);
                    implementation.emit(*state, callbacks[row], implementation.host_audio.data(),
                        size_t(count) * implementation.info.audio.samples_per_frame);
                    pause -= count;
                }
                if (!state->finished && state->metrics.frames < state->options.max_frames) {
                    ++state->segment;
                    implementation.start_segment(*state);
                } else state->finished = true;
            } else state->finished = true;
        }
        if (state->metrics.frames >= state->options.max_frames) state->finished = true;
        state->metrics.total_ms = milliseconds(state->start);
    }
}
std::shared_ptr<const Reference> ModelRunner::prepare_reference(ReferencePlan &plan) {
    ProgramSpec description{plan.program, plan.input, plan.outputs};
    return impl_->planner.finish_reference(plan, impl_->evaluate(description,
        impl_->workspace(description, ForwardContext::Plan)));
}
std::vector<int> ModelRunner::encode_audio(const std::vector<float> &mono) {
    auto &implementation = *impl_;
    if (mono.empty() || mono.size() > implementation.info.max_reference_samples)
        throw std::invalid_argument("reference audio is empty or exceeds its declared capacity");
    InputBatch input;
    input.limits = {1, int(mono.size()), int(mono.size())};
    input.sequence.shape = {1, int(mono.size()), int(mono.size())};
    input.uploads.push_back({"waveform", TensorView(const_cast<float *>(mono.data()), int(mono.size()), 1, Type::F32)});
    ProgramSpec description{implementation.definition.reference_program, input, {"codes"}};
    DeviceArena storage(implementation.runtime.memory, implementation.workspace(description, ForwardContext::Plan));
    Tensor::Scope scope(storage);
    ForwardContext context(1, int(mono.size()), int(mono.size()));
    NamedTensor outputs[] = {{"codes", {}}};
    run_forward(implementation.network, description.name, input, {}, {outputs, std::size(outputs)}, context,
                &implementation.runtime.memory);
    std::vector<int> result(outputs[0].value.numel());
    implementation.runtime.memory.download(result.data(), outputs[0].value.data, outputs[0].value.size());
    return result;
}
std::vector<float> ModelRunner::decode_codes(const std::vector<int> &codes) {
    auto &implementation = *impl_;
    const auto &format = implementation.info.audio;
    const auto &description = implementation.program(implementation.definition.audio_program, ForwardMode::Prefill);
    if (format.codebooks <= 0 || codes.empty() || codes.size() % format.codebooks ||
        codes.size() / format.codebooks > size_t(implementation.definition.max_decode_frames) ||
        std::any_of(codes.begin(), codes.end(), [&](int code) { return code < 0 || code >= format.codebook_size; }))
        throw std::invalid_argument("audio codes do not match the model's declared layout");
    const auto &specification = implementation.runtime.states.history_spec(description.history);
    auto allocate = [&] {
        SequenceState history(specification);
        Tensor::reserve(size_t(format.codebooks) * sizeof(int));
        Tensor::reserve(sizeof(int *));
        Tensor::reserve(sizeof(CacheSequence));
        Tensor::reserve(specification.history_bytes.size() * sizeof(ConvHistory));
    };
    const size_t history_bytes = Tensor::plan(allocate);
    auto input = description.input;
    input.limits = {1, 1, 1};
    input.sequence = {{}, {1, 1, 1}};
    input.parameters = {{"history_stride", 1}};
    ProgramSpec sizing{description.name, input, description.outputs};
    const size_t network_bytes = implementation.workspace(sizing, ForwardContext::Plan);
    DeviceArena storage(implementation.runtime.memory, history_bytes + network_bytes);
    Tensor::Scope allocation_scope(storage);
    SequenceState history(specification);
    history.reset(implementation.runtime.memory.stream);
    Buffer frame(size_t(format.codebooks) * sizeof(int)), pointers(sizeof(int *));
    Buffer cache(sizeof(CacheSequence)), histories(specification.history_bytes.size() * sizeof(ConvHistory));
    const int *frame_pointer = frame.ptr<int>();
    CacheSequence sequence{history.keys.data(), history.values.data(), history.layer_bytes, nullptr, history.position.ptr<int>()};
    std::vector<ConvHistory> descriptors;
    for (auto &buffer : history.history) descriptors.push_back({buffer.ptr<float>(), history.position.ptr<int>()});
    implementation.runtime.memory.upload(pointers.data(), &frame_pointer, sizeof(frame_pointer));
    implementation.runtime.memory.upload(cache.data(), &sequence, sizeof(sequence));
    implementation.runtime.memory.upload(histories.data(), descriptors.data(), descriptors.size() * sizeof(ConvHistory));
    const size_t frames = codes.size() / format.codebooks;
    std::vector<float> result(frames * format.samples_per_frame);
    for (size_t frame_index = 0; frame_index < frames; ++frame_index) {
        implementation.runtime.memory.upload(frame.data(), codes.data() + frame_index * format.codebooks, frame.size());
        Tensor::Scope scratch;
        ForwardContext context(1, 1, 1);
        context.batch.cache.layout.sequences = cache.ptr<CacheSequence>();
        const NamedTensor bindings[] = {{"code_rows", pointers.view(1, 1, Type::I64)},
            {"histories", histories.view(1, int(histories.size()), Type::I8)}};
        auto outputs = output_views(description.outputs);
        run_forward(implementation.network, description.name, input, {bindings, std::size(bindings)},
                    {outputs.data(), outputs.size()}, context);
        TensorInputs values(outputs.data(), outputs.size());
        const auto waveform = values["waveform"];
        if (waveform.type != Type::F32) throw std::invalid_argument("discrete audio decoder must return F32 samples");
        implementation.runtime.memory.download(result.data() + frame_index * format.samples_per_frame,
                                                waveform.data, waveform.size());
    }
    return result;
}
void ModelRunner::prepare_tasks(std::vector<PreparedTask> &tasks) {
    for (size_t first = 0; first < tasks.size();) {
        if (tasks[first].error) { ++first; continue; }
        const auto &prototype = tasks[first].program;
        size_t end = first + 1;
        std::vector<const InputBatch *> sources{&prototype.input};
        while (end < tasks.size() && !tasks[end].error) {
            const auto &program = tasks[end].program;
            const auto &parameters = program.input.parameters;
            const auto &expected = prototype.input.parameters;
            if (program.name != prototype.name || program.outputs != prototype.outputs ||
                program.input.fixed_position != prototype.input.fixed_position ||
                parameters.size() != expected.size() ||
                !std::equal(parameters.begin(), parameters.end(), expected.begin(),
                    [](const auto &left, const auto &right) { return left.name == right.name && left.value == right.value; }))
                break;
            sources.push_back(&program.input);
            ++end;
        }
        try {
            auto packed = impl_->pack(prototype.name, prototype.outputs, {sources.data(), sources.size()});
            end = first + packed.count;
            auto outputs = impl_->evaluate(packed.description, packed.workspace_bytes);
            std::vector<int> offsets(prototype.outputs.size());
            for (size_t field = 0; field < offsets.size(); ++field) {
                int64_t rows = 0;
                for (size_t index = first; index < end; ++index) rows += tasks[index].outputs[field].rows;
                if (rows != outputs.at(prototype.outputs[field]).size(0))
                    throw std::logic_error("preparation output does not match its row partitions");
            }
            for (size_t index = first; index < end; ++index) {
                HostResults result;
                for (size_t field = 0; field < offsets.size(); ++field) {
                    const auto &partition = tasks[index].outputs[field];
                    auto tensor = outputs.at(partition.name).narrow(0, offsets[field], partition.rows);
                    offsets[field] += partition.rows;
                    result.emplace(partition.name, std::move(tensor));
                }
                try { tasks[index].task->accept(std::move(result)); }
                catch (const DeviceError &) { throw; }
                catch (...) { tasks[index].error = std::current_exception(); }
            }
        } catch (const DeviceError &) { throw; }
        catch (const RequestOverloaded &) {
            end = first + 1;
            tasks[first].error = std::current_exception();
        } catch (...) {
            auto error = std::current_exception();
            for (size_t index = first; index < end; ++index) tasks[index].error = error;
        }
        first = end;
    }
}
} // namespace tts
