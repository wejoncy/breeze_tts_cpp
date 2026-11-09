#include "execution/model_runner.h"
#include "execution/input_staging.h"
#include "runtime/graph.h"
#include "ops/basic.h"
#include "ops/audio.h"
#include "ops/sampling.h"
#include "ops/attention.h"
#include "core/input_fields.h"
#include "support/profiler.h"
#include "support/init_log.h"
#include "support/log.h"
#include "runtime/memory_log.h"
#include <array>
#include <cmath>
#include <numeric>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

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
// Fixed Tensor storage and views keep graph addresses stable. Each program
// uploads one CPU snapshot; reuse is handled by the Tensor transfer backend.
struct PackedMetadata {
    Buffer device, host;
    explicit PackedMetadata(std::initializer_list<size_t> fields) {
        size_t bytes = 0;
        for (size_t field : fields) bytes += Tensor::aligned(field);
        device = Buffer(bytes);
        host = Buffer(bytes, Device::cpu());
    }
    TensorView take(size_t &offset, int rows, int columns, Type type) const {
        auto result = device.view(rows, columns, type, offset);
        offset += Tensor::aligned(result.size());
        return result;
    }
    template <class Value> Value *values(TensorView field) const {
        const auto offset = reinterpret_cast<uintptr_t>(field.data) - reinterpret_cast<uintptr_t>(device.data());
        return reinterpret_cast<Value *>(host.ptr<char>() + offset);
    }
    void upload() const {
        device.view(1, int(device.size()), Type::I8).copy_(host.view(1, int(host.size()), Type::I8), true);
    }
};
} // namespace
struct ModelRunner::Cache {
    std::shared_ptr<PagedKVCache::Reservation> capacity;
    std::shared_ptr<RequestSlots::Lease> slot;
    StatePool::Row &data;
    GenerationOptions options;
    std::unique_ptr<PagedKVCache::Lease> kv;
    Clock::time_point prefill_start = Clock::now();
    double prefill_ms = 0;
    int emitted_frames = 0;
    bool truncated = false;
    bool finished = false, is_prefilled = false;
    GenerationInput input;
    size_t segment = 0;
    int generated = 0, rendered = 0, audio_rendered = 0, windows = 0, segment_start_frames = 0;
    bool semantic_done = false;
    WindowSchedule schedule;
    std::vector<int16_t> pending;
    Cache(RunnerResources &resources, std::shared_ptr<PagedKVCache::Reservation> capacity,
          std::shared_ptr<RequestSlots::Lease> slot, GenerationInput value, GenerationOptions options)
        : capacity(std::move(capacity)), slot(std::move(slot)),
          data(*resources.states.rows().at(this->slot->index)), options(std::move(options)), input(std::move(value)) {}
    GenerationSegment &current() { return input.segments.at(segment); }
    const GenerationSegment &current() const { return input.segments.at(segment); }
};
struct ModelRunner::AudioDecodeCache {
    std::vector<int> codes;
    std::vector<float> audio;
    size_t frame = 0, workspace_bytes = 0;
    // Arena outlives all Tensor storage borrowed from it.
    std::unique_ptr<DeviceArena> arena;
    std::unique_ptr<SequenceState> history;
    Buffer token, pointer, cache, histories;
};
struct ModelRunner::Impl {
    using Cache = ModelRunner::Cache;
    const Model &network;
    RunnerResources &runtime;
    const ModelInfo &info;
    const InputDefinition definition;
    std::vector<ProgramSpec> programs;
    std::vector<Buffer> input_storage;
    std::unique_ptr<FlashDecodePlan> decode_plan;
    const int capacity, lane_capacity;
    const size_t history_bytes, pcm_bytes;
    PackedMetadata token_metadata, audio_metadata;
    TensorView samples, previous_codes, code_rows, owners, caches, counts, positions;
    TensorView audio_code_rows, audio_caches, histories;
    Buffer done, waveform_pcm;
    TensorView audio_output;
    std::vector<TensorView> audio_graph_outputs;
    int audio_stride = 0;
    size_t input_workspace = 0;
    size_t device_total_bytes = 0;
    size_t state_kv_bytes_per_row = 0;
    std::vector<int> host_layout;
    std::vector<int16_t> host_audio;
    struct RequestRecord {
        std::unique_ptr<RequestPlan> plan;
        size_t cache_pages = 0, cache_rows = 0, peak_cache_bytes = 0;
        bool cache_reported = false;
    };
    struct LiveCache { uint64_t request; std::shared_ptr<Cache> value; };
    // All execution state belongs to the runner. Scheduler sees IDs and records.
    std::unordered_map<uint64_t, RequestRecord> requests;
    std::unordered_map<uint64_t, LiveCache> states;
    ExecutionBatch scheduled;
    Completion completion;
    PreparationBatch preparation_batch;
    size_t preparation_first = 0, preparation_end = 0;
    std::vector<Cache *> prefill_rows;
    std::vector<std::vector<int16_t> *> audio_sinks;
    std::vector<Cache *> input_rows, step_rows, decode_rows;
    std::vector<size_t> audio_members;
    ProgramSpec audio_execution;
    size_t audio_total = 0, readback_bytes = 0;
    std::unique_ptr<DeviceArena> input_arena;
    Clock::time_point next_cache_log{};
    struct FrameRow {
        uint64_t id;
        std::shared_ptr<Cache> cache;
        bool decoded;
    };
    struct Frame {
        std::vector<FrameRow> rows;
        int slot = -1;
    } frame, submitted_frame;
    std::array<Buffer, 2> frame_storage;
    uint64_t engine_id = 0;
    ProgramSpec evaluation;
    HostResults evaluation_result;
    std::shared_ptr<AudioDecodeCache> audio_decode;
    void clear_control() noexcept {
        audio_decode.reset();
        evaluation_result.clear();
        evaluation = {};
    }
    bool has_work() const {
        return completion.executed || !preparation_batch.tasks.empty() || !evaluation.name.empty() || bool(audio_decode);
    }
    enum class Phase { Prefill, Decode, Audio, Frame, Preparation, Evaluation, Codes, Finish, Idle };
    Phase phase = Phase::Idle;
    void log_cache_usage(bool changed) {
        if (!log::enabled(TTS_LOG_INFO))
            return;
        const auto now = std::chrono::steady_clock::now();
        const bool periodic = now >= next_cache_log;
        if (!periodic && !changed)
            return;
        const auto usage = CacheUsage{runtime.kv.storage_page_bytes(), device_total_bytes,
            state_kv_bytes_per_row, runtime.kv.capacity() - runtime.kv.available(),
            runtime.kv.capacity(), capacity};
        constexpr double mib = 1024.0 * 1024.0;
        const auto percent = [](size_t used, size_t total) {
            return total ? 100.0 * double(used) / double(total) : 0.0;
        };
        for (auto &entry : requests) {
            entry.second.cache_pages = 0;
            entry.second.cache_rows = 0;
        }
        for (const auto &entry : states) {
            auto &request = requests.at(entry.second.request);
            ++request.cache_rows;
            if (const auto &cache = entry.second.value->kv)
                request.cache_pages += cache->pages().size();
        }
        bool admitted = false;
        for (auto &entry : requests) {
            auto &request = entry.second;
            const size_t used = request.cache_pages * usage.page_bytes +
                request.cache_rows * usage.state_kv_bytes_per_row;
            request.peak_cache_bytes = std::max(request.peak_cache_bytes, used);
            const bool first = !request.cache_reported;
            admitted |= first;
            if ((first || periodic) && log::enabled(TTS_LOG_DEBUG)) {
                log::Scope scope(engine_id, entry.first);
                TTS_LOG(TTS_LOG_DEBUG, "request.kv",
                        "{} used={:.0f} MiB paged_pages={} observed_peak={:.0f} MiB device_used={:.2f}%",
                        first ? "admitted" : "active", used / mib, request.cache_pages,
                        request.peak_cache_bytes / mib,
                        percent(used, usage.device_bytes));
            }
            request.cache_reported = true;
        }
        if (periodic || admitted || changed) {
            const size_t used = size_t(usage.used_pages) * usage.page_bytes +
                states.size() * usage.state_kv_bytes_per_row;
            const size_t paged_backed = size_t(usage.backed_pages) * usage.page_bytes;
            const size_t backed = paged_backed + size_t(usage.state_rows) * usage.state_kv_bytes_per_row;
            TTS_LOG(TTS_LOG_INFO, "engine.kv", "KV cache usage={:.1f}%", percent(used, backed));
            next_cache_log = now + std::chrono::seconds(1);
        }
    }
    void release_request(uint64_t id) {
        const auto found = requests.find(id);
        if (found == requests.end())
            return;
        log::Scope scope(engine_id, id);
        TTS_LOG(TTS_LOG_DEBUG, "request.kv", "released observed_peak={:.0f} MiB",
                found->second.peak_cache_bytes / (1024.0 * 1024.0));
        requests.erase(found);
    }
    void clear_requests() {
        const bool changed = !states.empty() || !requests.empty();
        states.clear();
        requests.clear();
        if (changed)
            log_cache_usage(true);
    }
    int available(const Cache &state) const {
        return std::min(state.options.max_frames - state.segment_start_frames,
            state.input.rate.frames(state.generated, state.options.speed_factor));
    }
    int ready(const Cache &state) const {
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
    bool audio_pending(const Cache &state) const {
        return state.input.windows.mode == SynthesisMode::Offline && state.semantic_done &&
            state.rendered == available(state) && state.audio_rendered < state.rendered;
    }
    void append_pcm(Cache &state, std::vector<int16_t> &output, const int16_t *audio, size_t count) const {
        if (!count || state.finished) return;
        const int frame_samples = info.audio.samples_per_frame;
        if (count % frame_samples || count / frame_samples > size_t(state.options.max_frames - state.emitted_frames))
            throw std::logic_error("audio delivery exceeds the admitted frame limit");
        if (count > output.capacity() - output.size())
            throw std::logic_error("audio step exceeds initialized output capacity");
        output.insert(output.end(), audio, audio + count);
        state.emitted_frames += int(count / frame_samples);
    }
    ProgramSpec audio_description(const std::vector<Cache *> &rows) const {
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
        append("generated", [](const Cache &state) { return state.generated; });
        append("rendered", [](const Cache &state) { return state.rendered; });
        append("audio_rendered", [](const Cache &state) { return state.audio_rendered; });
        append("available_frames", [&](const Cache &state) { return available(state); });
        append("frames", [&](const Cache &state) { return ready(state); });
        append("history_frames", [](const Cache &state) { return std::min(state.rendered, state.input.windows.left); });
        append("future_frames", [&](const Cache &state) {
            return std::min(state.input.windows.lookahead(state.rendered), available(state) - state.rendered - ready(state));
        });
        append("seed", [](const Cache &state) { return state.options.seed; });
        append("windows", [](const Cache &state) { return uint64_t(state.windows); });
        return result;
    }
    struct StateBindings {
        std::vector<std::vector<TensorView>> values;
        std::vector<NamedValue<ArrayView<const TensorView>>> fields;
        void bind(ForwardContext &context) const { context.state = {fields.data(), fields.size()}; }
    };
    std::vector<Cache *> active_audio;
    StateBindings state_bindings(const std::vector<Cache *> &rows, int padded) const {
        StateBindings result;
        const auto &prototype = runtime.states.rows().front()->fields();
        result.values.reserve(prototype.size());
        result.fields.reserve(prototype.size());
        for (const auto &field : prototype) {
            result.values.emplace_back();
            auto &values = result.values.back();
            values.reserve(padded);
            for (int row = 0; row < padded; ++row) {
                const auto &source = rows.empty() ? *runtime.states.rows().at(std::min<size_t>(row, capacity - 1))
                    : rows.at(std::min<size_t>(row, rows.size() - 1))->data;
                values.push_back(source.view(field.first));
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
    std::array<NamedTensor, 8> bindings(int rows, int lanes, bool audio = false) const {
        return {{{"sampling", samples.slice(0, rows)},
                 {"previous_codes", previous_codes.slice(0, rows)},
                 {"code_rows", (audio ? audio_code_rows : code_rows).slice(0, rows)},
                 {"owner_rows", owners.slice(0, lanes)},
                 {"done", done.view(rows, 1, Type::I32)},
                 {"audio_positions", positions.slice(0, rows)},
                 {"histories", histories},
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
        auto arguments = bindings(capacity, lane_capacity, description.name == definition.audio_program);
        auto outputs = output_views(description.outputs);
        run_forward(network, description.name, description.input,
            {arguments.data(), arguments.size()}, {outputs.data(), outputs.size()}, context);
        size_t total = 0;
        for (const auto &field : outputs) if (field.value.device >= 0)
            total += Tensor::aligned(field.value.size());
        readback_bytes = std::max(readback_bytes, total);
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
    void invoke(const ProgramSpec &description, int rows, int lanes, int partitions = 1,
                ForwardContext::Mode mode = ForwardContext::Execute, bool replay = true,
                const NamedValue<ForwardShape> *example = nullptr) {
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
        context.example = example;
        if (description.replay == ProgramSpec::Replay::Regions) {
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
        // Token decode can be submitted while the previous audio readback still
        // owns active_audio. Token programs use their normal slot bindings.
        const std::vector<Cache *> no_audio;
        auto state = state_bindings(description.name == definition.main_program ? no_audio : active_audio, rows);
        state.bind(context);
        auto arguments = bindings(rows, lanes, description.name == definition.audio_program);
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
            if (selected != outputs.end()) {
                // This view borrows the graph workspace. launch_frame copies
                // it directly into the batch result slot on the same stream,
                // before the next token program can reuse that workspace.
                audio_output = selected->value;
                if (audio_output.size() > pcm_bytes) throw std::logic_error("audio output exceeds storage");
            }
            else {
                auto waveform = result["waveform"];
                if (waveform.size() && waveform.numel() * sizeof(int16_t) > waveform_pcm.size())
                    throw std::logic_error("waveform output exceeds PCM storage");
                if (waveform.numel()) tts::pcm(waveform, waveform_pcm.ptr<int16_t>());
                audio_output = waveform_pcm.view(waveform.rows, waveform.cols, Type::I16);
            }
            audio_stride = int(audio_output.numel() / rows);
            if (description.replay == ProgramSpec::Replay::Batch)
                audio_graph_outputs.at(rows) = audio_output;
        }
    }
    void map(const std::vector<Cache *> &rows, const std::vector<int> *cursors = nullptr) {
        auto *sampling = token_metadata.values<SamplingRow>(samples);
        auto *previous = token_metadata.values<const int *>(previous_codes);
        auto *codes = token_metadata.values<const int *>(code_rows);
        auto *sequences = token_metadata.values<CacheSequence>(caches);
        auto *membership = token_metadata.values<int>(owners);
        auto *generated = token_metadata.values<int>(counts);
        auto *relative = token_metadata.values<int>(positions);
        std::fill_n(sampling, capacity, SamplingRow{});
        std::fill_n(previous, capacity, nullptr);
        std::fill_n(codes, capacity, nullptr);
        std::fill_n(sequences, lane_capacity, CacheSequence{});
        std::fill_n(membership, lane_capacity, -1);
        std::fill_n(generated, capacity, 0);
        std::fill_n(relative, capacity, 0);
        int lane_index = 0;
        for (size_t row = 0; row < rows.size(); ++row) {
            auto &state = *rows[row];
            const int cursor = cursors ? cursors->at(row) : state.generated;
            auto &data = state.data;
            const size_t token_bytes = size_t(definition.codebooks) * sizeof(int);
            const bool history = data["codes"].size() > token_bytes;
            auto *tokens = data["codes"].ptr<int>();
            previous[row] = tokens + size_t(history ? std::max(0, cursor - 1) : 0) * definition.codebooks;
            codes[row] = tokens + size_t(history ? cursor : 0) * definition.codebooks;
            auto &sample = sampling[row];
            sample.codes = const_cast<int *>(codes[row]);
            sample.done = data["done"].ptr<int>();
            sample.seen = data["seen"].ptr<int>();
            sample.rng = data["rng"].ptr<SamplingRng>();
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
            generated[row] = cursor;
            relative[row] = state.current().position_offset + std::max(0, cursor - 1);
            auto cache = state.kv->view();
            for (int branch = 0; branch < sample.branches; ++branch) {
                sample.logits[branch] = lane_index;
                membership[lane_index] = int(row);
                sequences[lane_index++] = {cache.keys.data, cache.values.data,
                    size_t(cache.rows_per_layer) * cache.keys.cols * bytes(cache.keys.type),
                    cache.layout.block_table + branch * cache.layout.block_stride,
                    data["positions"].ptr<int>() + branch};
            }
        }
        token_metadata.upload();
    }
    void sampled(const std::vector<Cache *> &rows, TensorView values) {
        if (values.type != Type::I32 || values.numel() != rows.size())
            throw std::logic_error("sample completion does not match the forward rows");
        for (size_t row = 0; row < rows.size(); ++row) {
            auto &state = *rows[row];
            if (values.ptr<int>()[row]) state.semantic_done = true;
            else if (++state.generated == state.current().max_tokens) {
                state.semantic_done = true;
                state.truncated = true;
            }
        }
    }
    void launch_decode(const std::vector<Cache *> &decode, const std::vector<int> *cursors = nullptr) {
        int lanes = 0, key_length = 0;
        bool enabled = true;
        for (size_t row = 0; row < decode.size(); ++row) {
            auto *state = decode[row];
            const int cursor = cursors ? cursors->at(row) : state->generated;
            auto lengths = state->current().lengths;
            for (auto &length : lengths) length += cursor;
            state->kv->ensure(lengths);
            lanes += int(state->current().lengths.size());
            for (int length : state->current().lengths)
                key_length = std::max(key_length, length + cursor);
            enabled &= state->options.use_cuda_graph;
        }
        auto shape = runtime.shapes.select(int(decode.size()), lanes, definition.max_branches);
        if (!shape.first) shape = {int(decode.size()), lanes};
        map(decode, cursors);
        const int partitions = decode_plan
            ? decode_plan->select(shape.second, definition.query_heads, key_length) : 1;
        const auto &description = program(definition.main_program, ForwardMode::Decode);
        runtime.execution.graphs().run(description.name.c_str(), {shape.first, shape.second, partitions}, enabled,
            [&] { invoke(description, shape.first, shape.second, partitions); });
    }
    void launch_audio(const std::vector<Cache *> &audio, const ProgramSpec &description) {
        int padded = description.replay == ProgramSpec::Replay::None ? int(audio.size())
            : runtime.shapes.select(int(audio.size()));
        if (!padded) padded = int(audio.size());
        auto *codes = audio_metadata.values<const int *>(audio_code_rows);
        auto *sequences = audio_metadata.values<CacheSequence>(audio_caches);
        const size_t layers = description.history.empty() ? 0
            : runtime.states.history_spec(description.history).history_bytes.size();
        auto *history = audio_metadata.values<ConvHistory>(histories);
        std::fill_n(codes, capacity, nullptr);
        std::fill_n(sequences, capacity, CacheSequence{});
        std::fill_n(history, history_bytes / sizeof(ConvHistory), ConvHistory{});
        bool enabled = true;
        for (size_t row = 0; row < audio.size(); ++row) {
            auto &cache = *audio[row];
            const size_t books = definition.codebooks;
            const bool retained = cache.data["codes"].size() > books * sizeof(int);
            codes[row] = cache.data["codes"].ptr<int>() + size_t(retained ? cache.generated - 1 : 0) * books;
            enabled &= cache.options.use_cuda_graph;
            if (!description.history.empty()) {
                auto &sequence = cache.data.history(description.history);
                sequences[row] = {sequence.keys.data(), sequence.values.data(), sequence.layer_bytes,
                                  nullptr, sequence.position.ptr<int>()};
                for (size_t layer = 0; layer < layers; ++layer)
                    history[layer * capacity + row] = {sequence.history[layer].ptr<float>(), sequence.position.ptr<int>()};
            }
        }
        audio_metadata.upload();
        active_audio = audio;
        const bool replayed = runtime.execution.graphs().run(description.name.c_str(), {padded}, enabled,
            [&] { invoke(description, padded, padded, 1, ForwardContext::Execute, enabled); });
        if (replayed) {
            audio_output = audio_graph_outputs.at(padded);
            audio_stride = int(audio_output.numel() / padded);
        }
    }
    bool framewise() const {
        const auto &audio = program(definition.audio_program, ForwardMode::Prefill);
        return definition.max_audio_frames == 1 && audio.replay == ProgramSpec::Replay::Batch &&
            !audio.history.empty() && std::find(audio.outputs.begin(), audio.outputs.end(), "pcm") != audio.outputs.end();
    }
    bool framewise(const Cache &cache) const {
        const auto &windows = cache.input.windows;
        return framewise() && cache.input.segments.size() == 1 &&
            cache.input.rate.semantic_repeat == 1 && cache.input.rate.feature_repeat == 1 &&
            cache.options.speed_factor == 1 && windows.mode != SynthesisMode::Offline &&
            windows.core == 1 && !windows.future && !windows.holdback && !windows.fade && !windows.left &&
            cache.current().audio.bindings.empty() && cache.current().audio.uploads.empty() &&
            cache.data["codes"].size() == size_t(definition.codebooks) * sizeof(int);
    }
    Frame launch_frame(const ExecutionBatch &batch, const Frame *previous = nullptr) {
        Frame packet;
        packet.rows.reserve(batch.rows.size());
        std::vector<FrameRow> pending_audio;
        std::vector<int> cursors;
        cursors.reserve(batch.rows.size());
        for (const auto &row : batch.rows) {
            const auto found = states.find(row.id);
            if (found == states.end()) continue;
            const auto &handle = found->second.value;
            auto &cache = *handle;
            if (!cache.is_prefilled || cache.finished) continue;
            int cursor = cache.generated;
            bool decoded = cache.generated == cache.rendered;
            if (previous) {
                auto old = std::find_if(previous->rows.begin(), previous->rows.end(),
                    [&](const FrameRow &entry) { return entry.cache == handle; });
                if (old == previous->rows.end()) continue;
                decoded = true;
                cursor += old->decoded ? 1 : 0;
            }
            if (decoded && (cache.semantic_done || cursor >= cache.current().max_tokens)) continue;
            if (decoded) cursors.push_back(cursor);
            FrameRow entry{row.id, handle, decoded};
            (decoded ? packet.rows : pending_audio).push_back(std::move(entry));
        }
        const size_t decoded = packet.rows.size();
        packet.rows.insert(packet.rows.end(), pending_audio.begin(), pending_audio.end());
        if (packet.rows.empty()) return packet;
        packet.slot = previous ? 1 - previous->slot : 0;
        auto &storage = frame_storage[packet.slot];
        std::vector<Cache *> decode;
        decode.reserve(decoded);
        for (size_t row = 0; row < decoded; ++row) {
            decode.push_back(packet.rows[row].cache.get());
        }
        if (decoded) {
            launch_decode(decode, &cursors);
            storage.view(int(decoded), 1, Type::I32).copy_(done.view(int(decoded), 1, Type::I32), true);
        }
        if (decoded < packet.rows.size())
            ops::zero(storage.view(int(packet.rows.size() - decoded), 1, Type::I32, decoded * sizeof(int)));
        active_audio.clear(); active_audio.reserve(packet.rows.size());
        for (const auto &row : packet.rows) active_audio.push_back(row.cache.get());
        launch_audio(active_audio, program(definition.audio_program, ForwardMode::Prefill));
        const int samples = info.audio.samples_per_frame;
        storage.view(int(packet.rows.size()), samples, Type::I16, Tensor::aligned(done.size()))
            .copy_(audio_output.slice(0, int(packet.rows.size())), true);
        return packet;
    }
    void start_segment(Cache &state) {
        auto &input = state.current();
        state.data.reset_host();
        state.kv.reset();
        state.kv = std::make_unique<PagedKVCache::Lease>(runtime.kv, input.lengths, input.capacities,
                                                          state.data["table"].ptr<int>(), state.capacity.get());
        for (auto name : {"codes", "done", "seen"}) {
            auto tensor = state.data[name];
            if (tensor.size()) ops::zero(tensor);
        }
        state.options.seed = input.seed;
        SamplingRng random{input.seed, 0};
        state.data["rng"].copy_({&random, 1, 2, Type::I64}, true);
        state.data["positions"].slice(0, int(input.lengths.size()))
            .copy_({input.lengths.data(), int(input.lengths.size()), 1, Type::I32}, true);
        for (const auto &field : input.initializers) {
            auto destination = state.data[field.name];
            if (field.value.size() > destination.size()) throw std::invalid_argument("initial state exceeds its declared capacity");
            TensorView(destination.data, field.value.rows, field.value.cols, field.value.type, 0, destination.device)
                .copy_(field.value, true);
        }
        for (const auto &address : input.input.addresses) {
            auto values = std::make_shared<Tensor>(Tensor::zeros({int(address.offsets.size()), 1}, Type::I64));
            auto *pointers = values->view().ptr<uint64_t>();
            auto buffer = state.data[address.state];
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
        state.segment_start_frames = state.emitted_frames;
        const double now = std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
        state.schedule = WindowSchedule(state.input.windows.core, state.input.windows.maximum, now);
        state.pending.clear();
        state.pending.reserve(size_t(state.input.windows.fade) * info.audio.samples_per_frame);
        state.is_prefilled = state.semantic_done = false;
    }
    Impl(const Model &model, RunnerResources &resources, const InputDefinition &inputs,
          const ModelInfo &metadata, size_t runtime_bytes)
        : network(model), runtime(resources), info(metadata), definition(inputs), programs(definition.programs),
          capacity(resources.batch), lane_capacity(capacity * definition.max_branches),
          history_bytes([&] {
              size_t maximum = 0;
              for (const auto &description : programs)
                  if (!description.history.empty()) maximum = std::max(maximum,
                      resources.states.rows().front()->history(description.history).history.size());
              return maximum * capacity * sizeof(ConvHistory);
          }()),
          pcm_bytes(size_t(capacity) * definition.max_audio_frames * info.audio.samples_per_frame * sizeof(int16_t)),
          token_metadata({size_t(capacity) * sizeof(SamplingRow), size_t(capacity) * sizeof(void *),
              size_t(capacity) * sizeof(void *), size_t(lane_capacity) * sizeof(int),
              size_t(lane_capacity) * sizeof(CacheSequence), size_t(capacity) * sizeof(int),
              size_t(capacity) * sizeof(int)}),
          audio_metadata({size_t(capacity) * sizeof(void *), size_t(capacity) * sizeof(CacheSequence), history_bytes}),
          done(size_t(capacity) * sizeof(int)),
          waveform_pcm([&] {
              const auto &outputs = program(definition.audio_program, ForwardMode::Prefill).outputs;
              return std::find(outputs.begin(), outputs.end(), "pcm") == outputs.end() ? pcm_bytes : 0;
          }()), audio_graph_outputs(size_t(capacity) + 1),
          host_layout(size_t(capacity) * 2), host_audio(pcm_bytes / sizeof(int16_t)) {
        size_t offset = 0;
        samples = token_metadata.take(offset, capacity, sizeof(SamplingRow), Type::I8);
        previous_codes = token_metadata.take(offset, capacity, 1, Type::I64);
        code_rows = token_metadata.take(offset, capacity, 1, Type::I64);
        owners = token_metadata.take(offset, lane_capacity, 1, Type::I32);
        caches = token_metadata.take(offset, lane_capacity, sizeof(CacheSequence), Type::I8);
        counts = token_metadata.take(offset, capacity, 1, Type::I32);
        positions = token_metadata.take(offset, capacity, 1, Type::I32);
        offset = 0;
        audio_code_rows = audio_metadata.take(offset, capacity, 1, Type::I64);
        audio_caches = audio_metadata.take(offset, capacity, sizeof(CacheSequence), Type::I8);
        histories = audio_metadata.take(offset, 1, int(history_bytes), Type::I8);
        readback_bytes = pcm_bytes;
        if (framewise()) {
            readback_bytes = Tensor::aligned(done.size()) + Tensor::aligned(pcm_bytes);
            for (auto &storage : frame_storage) storage = Buffer(readback_bytes);
        }
        log::InitPhase phase("execution storage");
        size_t history_bytes_per_row = 0;
        for (const auto &history : runtime.states.layout().histories) {
            state_kv_bytes_per_row += history.second.kv_bytes * 2;
            for (size_t bytes : history.second.history_bytes) history_bytes_per_row += bytes;
        }
        TTS_LOG(TTS_LOG_DEBUG, "engine.init",
                "persistent_state rows={} kv_backed={:.0f} MiB convolution_history={:.0f} MiB",
                capacity, state_kv_bytes_per_row * capacity / (1024.0 * 1024.0),
                history_bytes_per_row * capacity / (1024.0 * 1024.0));
        if (definition.query_heads > 0) {
            decode_plan = std::make_unique<FlashDecodePlan>(std::max(1,
                ops::decode_capacity(runtime.memory.device, definition.head_dimensions, definition.type)),
                runtime.context);
            ops::reserve_decode(runtime.memory.device, definition.head_dimensions, definition.type);
        }
        for (auto *buffer : {&token_metadata.device, &audio_metadata.device, &done}) runtime.memory.zero(*buffer);
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
        phase.next("workspace planning");
        size_t retained = 0, preparation = 0;
        for (const auto &description : programs) {
            preparation = std::max(preparation, workspace(description, ForwardContext::Prepare));
            const size_t bytes = workspace(description, ForwardContext::Plan);
            if (description.replay != ProgramSpec::Replay::None || description.name == definition.audio_program)
                retained = std::max(retained, bytes);
            if (description.replay == ProgramSpec::Replay::None && description.name != definition.audio_program)
                input_workspace = std::max(input_workspace, bytes);
        }
        TTS_LOG(TTS_LOG_DEBUG, "engine.init",
                "workspaces retained={:.0f} MiB preparation={:.0f} MiB input_max={:.0f} MiB",
                retained / (1024.0 * 1024.0), preparation / (1024.0 * 1024.0),
                input_workspace / (1024.0 * 1024.0));
        runtime.execution.prepare(retained);
        phase.next("operator preparation");
        {
            DeviceArena storage(runtime.memory, preparation);
            for (const auto &description : programs) {
                Tensor::Scope scope(storage);
                prepare_forward(description, ForwardContext::Prepare);
            }
        }
        phase.next("CUDA graph capture");
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
                std::vector<NamedValue<ForwardShape>> examples;
                for (const auto &region : description.regions) {
                    auto lengths = region.lengths;
                    lengths.erase(std::remove_if(lengths.begin(), lengths.end(),
                        [&](int length) { return length < 1 || length > region.maximum_time; }), lengths.end());
                    std::sort(lengths.begin(), lengths.end(), std::greater<int>());
                    lengths.erase(std::unique(lengths.begin(), lengths.end()), lengths.end());
                    const auto &batches = runtime.shapes.rows();
                    for (auto batch = batches.rbegin(); batch != batches.rend(); ++batch)
                        for (int length : lengths) {
                            if (*batch > INT32_MAX / length)
                                throw std::invalid_argument("graph shape exceeds tensor dimensions");
                            examples.push_back({region.name, {*batch, length, *batch * length}});
                        }
                }
                GraphCaptureProgress progress(description.name.c_str(), examples.size());
                for (const auto &example : examples) {
                    invoke(description, capacity, capacity, 1, ForwardContext::Capture, true, &example);
                    progress.advance();
                }
            }
        }
        phase.next("execution plan finalization");
        runtime.execution.seal();
        ops::seal(runtime.memory.device);
        runtime.memory.trim();
        size_t transfers = pcm_bytes;
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
        for (auto *buffer : {&token_metadata.device, &audio_metadata.device, &done})
            transfers += Tensor::aligned(buffer->size());
        for (const auto &row : runtime.states.rows())
            for (const auto &field : row->fields())
                if (!field.second.device().is_cpu()) transfers += Tensor::aligned(field.second.nbytes());
        phase.next("host transfer reservation");
        // At most the current audio stage and one submitted successor retain
        // upload snapshots. Neither may recycle the other's pending H2D data.
        if (transfers > SIZE_MAX / 2) throw std::overflow_error("host transfer reservation");
        const size_t transfer_capacity = transfers * 2;
        TTS_LOG(TTS_LOG_DEBUG, "engine.init",
                "host_transfer_staging={:.0f} MiB pinned_tensor_pool={:.0f} MiB",
                Tensor::aligned(transfer_capacity) / (1024.0 * 1024.0),
                (Tensor::aligned(transfer_capacity) + Tensor::aligned(readback_bytes)) / (1024.0 * 1024.0));
        runtime.memory.reserve_transfers(transfer_capacity, readback_bytes);
        phase.next("runtime memory reservation");
        const size_t available = runtime.memory.available_bytes();
        const size_t budget = runtime_bytes ? runtime_bytes
            : std::min(input_workspace + runtime.kv.max_storage_bytes(), available - available / 10);
        TTS_LOG(TTS_LOG_INFO, "engine.init",
                "runtime_budget={:.0f} MiB kv_max={:.0f} MiB",
                budget / (1024.0 * 1024.0), runtime.kv.max_storage_bytes() / (1024.0 * 1024.0));
        if (!budget || budget > available) throw DeviceOutOfMemory("runtime reservation exceeds the available memory budget");
        runtime.memory.reserve_runtime(budget);
        device_total_bytes = log_device_memory(runtime.memory, "ready");
    }

};
ModelRunner::ModelRunner(const Model &model, RunnerResources &resources, const InputDefinition &inputs,
                         const ModelInfo &info, size_t runtime_bytes)
    : impl_(std::make_unique<Impl>(model, resources, inputs, info, runtime_bytes)) {}
ModelRunner::~ModelRunner() = default;
size_t ModelRunner::max_readback_bytes() const { return impl_->readback_bytes; }
void ModelRunner::set_engine_id(uint64_t id) { impl_->engine_id = id; }
bool ModelRunner::finished() const { return impl_->phase == Impl::Phase::Idle; }
bool ModelRunner::has_batch() const { return impl_->completion.executed; }
bool ModelRunner::preparing() const { return !impl_->preparation_batch.tasks.empty(); }
bool ModelRunner::has_control() const { return !impl_->evaluation.name.empty() || impl_->audio_decode; }
bool ModelRunner::has_submitted() const { return impl_->submitted_frame.slot >= 0; }
bool ModelRunner::can_submit_next() const {
    return impl_->completion.executed && impl_->phase == Impl::Phase::Frame && !has_submitted();
}
void ModelRunner::abort() {
    impl_->runtime.memory.sync();
    impl_->submitted_frame = {};
    impl_->frame = {};
    impl_->clear_control();
    impl_->input_arena.reset();
    impl_->clear_requests();
    impl_->phase = Impl::Phase::Idle;
    impl_->completion = {};
    impl_->scheduled = {};
    impl_->preparation_batch = {};
}
bool ModelRunner::submit_next(const ExecutionBatch &batch) {
    auto &i = *impl_;
    if (has_submitted()) throw std::logic_error("a successor is already submitted");
    if (!can_submit_next() || batch.rows.empty() ||
        !batch.requests.empty() || !batch.release.empty() || !batch.release_requests.empty()) return false;
    for (const auto &row : batch.rows) {
        auto found = i.states.find(row.id);
        if (row.create || (found != i.states.end() && !i.framewise(*found->second.value))) return false;
    }
    Range trace("tts.runner.submit_next");
    i.submitted_frame = i.launch_frame(batch, &i.frame);
    return has_submitted();
}
ForwardOutputs ModelRunner::forward(ExecutionBatch batch) {
    auto &i = *impl_;
    if (i.has_work()) throw std::logic_error("runner already has scheduled work");
    const bool submitted = has_submitted();
    if (submitted) {
        for (const auto &row : i.submitted_frame.rows)
            if (std::none_of(batch.rows.begin(), batch.rows.end(),
                [&](const ExecutionRow &entry) { return entry.id == row.id; }))
                throw std::logic_error("submitted batch changed");
        i.frame = std::move(i.submitted_frame);
        i.submitted_frame = {};
    }
    i.scheduled = std::move(batch);
    i.step_rows.clear();
    i.completion = std::move(i.scheduled.output);
    auto &result = i.completion;
    result.valid_rows = 0;
    result.fatal = {};
    result.decoded_rows = 0;
    result.executed = true;
    for (auto id : i.scheduled.release) {
        auto found = i.states.find(id);
        if (found != i.states.end()) { found->second.value->finished = true; i.states.erase(found); }
    }
    for (auto id : i.scheduled.release_requests) i.release_request(id);
    for (auto &request : i.scheduled.requests)
        i.requests.emplace(request.id, Impl::RequestRecord{std::move(request.plan)});
    std::unordered_map<uint64_t, std::exception_ptr> rejected;
    i.scheduled.rows.erase(std::remove_if(i.scheduled.rows.begin(), i.scheduled.rows.end(),
        [&](const ExecutionRow &row) { return !row.create && !i.states.count(row.id); }), i.scheduled.rows.end());
    if (i.scheduled.rows.size() > result.rows.size())
        throw std::logic_error("batch exceeds initialized completion capacity");
    for (const auto &row : i.scheduled.rows) {
        auto &out = result.rows[result.valid_rows++];
        out.id = row.id;
        out.stats = {};
        out.error = {};
        out.done = out.prefilled = false;
        out.pcm.clear();
        if (rejected.count(row.request)) continue;
        if (!i.states.count(row.id)) {
            try {
                log::Scope scope(i.engine_id, row.request);
                auto &request = i.requests.at(row.request);
                auto state = prepare(*request.plan, row);
                i.states.emplace(row.id, Impl::LiveCache{row.request, std::move(state)});
            } catch (const DeviceError &) { throw; }
            catch (...) { rejected[row.request] = std::current_exception(); }
        }
    }
    i.prefill_rows.clear();
    for (size_t index = 0; index < i.scheduled.rows.size(); ++index) {
        const auto &row = i.scheduled.rows[index];
        auto &out = result.rows[index];
        if (auto error = rejected.find(row.request); error != rejected.end()) {
            out.error = error->second;
            out.done = true;
        }
        auto found = i.states.find(row.id);
        if (!out.done && found != i.states.end() &&
            !found->second.value->is_prefilled && !found->second.value->finished)
            i.prefill_rows.push_back(found->second.value.get());
    }
    i.phase = i.prefill_rows.empty() ? Impl::Phase::Decode : Impl::Phase::Prefill;
    if (submitted) {
        decode_forward();
        return frame_outputs();
    }
    return forward();
}

ForwardOutputs ModelRunner::forward(PreparationBatch batch) {
    auto &i = *impl_;
    if (i.has_work()) throw std::logic_error("runner already has scheduled work");
    if (batch.tasks.empty()) throw std::invalid_argument("empty preparation batch");
    i.preparation_batch = std::move(batch);
    i.preparation_first = i.preparation_end = 0;
    i.phase = Impl::Phase::Preparation;
    return forward();
}

ForwardOutputs ModelRunner::forward() {
    auto &i = *impl_;
    switch (i.phase) {
    case Impl::Phase::Prefill: return prefill_forward(i.prefill_rows);
    case Impl::Phase::Decode: return decode_forward();
    case Impl::Phase::Audio: return audio_forward();
    case Impl::Phase::Frame: return frame_outputs();
    case Impl::Phase::Preparation: return preparation_forward();
    case Impl::Phase::Evaluation: {
        const auto &description = i.evaluation;
        i.input_arena = std::make_unique<DeviceArena>(i.runtime.memory,
            i.workspace(description, ForwardContext::Plan));
        Tensor::Scope scope(*i.input_arena);
        const auto &input = description.input;
        ForwardContext context(input.limits.batch, input.limits.time, input.limits.rows);
        auto outputs = output_views(description.outputs);
        run_forward(i.network, description.name, input, {}, {outputs.data(), outputs.size()}, context);
        return {std::move(outputs)};
    }
    case Impl::Phase::Codes: return decode_codes_forward();
    case Impl::Phase::Finish:
        finish_step();
        i.log_cache_usage(!i.scheduled.release.empty() || !i.scheduled.release_requests.empty());
        for (size_t index = 0; index < i.completion.valid_rows; ++index) {
            auto &out = i.completion.rows[index];
            auto found = i.states.find(out.id);
            if (found == i.states.end()) continue;
            auto &state = *found->second.value;
            out.stats.prefill_ms = state.prefill_ms;
            out.stats.frames = state.emitted_frames;
            out.stats.audio_seconds = double(state.emitted_frames) * i.info.audio.samples_per_frame / i.info.audio.sample_rate;
            out.stats.truncated = state.truncated;
            out.prefilled = state.is_prefilled;
            out.done |= state.finished;
            if (out.done) i.states.erase(found);
        }
        i.phase = Impl::Phase::Idle;
        return {};
    case Impl::Phase::Idle: return {};
    }
    throw std::logic_error("unknown forward phase");
}

void ModelRunner::commit(const SampledOutputs &sampled) {
    auto &i = *impl_;
    TensorInputs values(sampled.values.data(), sampled.values.size());
    switch (i.phase) {
    case Impl::Phase::Prefill:
        i.sampled(i.input_rows, values["done"]);
        for (auto *state : i.input_rows) {
            state->is_prefilled = true;
            state->prefill_ms = milliseconds(state->prefill_start);
        }
        i.input_arena.reset();
        // Return completed prefills to Scheduler before starting another model
        // step. Newly prepared waiting rows may now outrank existing decoders.
        if (std::none_of(i.prefill_rows.begin(), i.prefill_rows.end(),
            [](const auto *state) { return !state->is_prefilled && !state->finished; })) i.phase = Impl::Phase::Finish;
        break;
    case Impl::Phase::Decode:
        i.sampled(i.decode_rows, values["done"]);
        i.phase = Impl::Phase::Audio;
        break;
    case Impl::Phase::Audio: commit_audio(sampled); break;
    case Impl::Phase::Frame: commit_frame(sampled); break;
    case Impl::Phase::Preparation: commit_preparation(sampled); break;
    case Impl::Phase::Evaluation:
        for (const auto &field : sampled.values)
            i.evaluation_result.emplace(field.name, Tensor(field.value));
        i.input_arena.reset();
        i.phase = Impl::Phase::Idle;
        break;
    case Impl::Phase::Codes: {
        const auto waveform = values["waveform"];
        const size_t samples = size_t(i.info.audio.samples_per_frame);
        if (waveform.type != Type::F32 || waveform.numel() != samples)
            throw std::logic_error("discrete audio decoder must return one F32 frame");
        auto &cache = *i.audio_decode;
        std::copy_n(waveform.ptr<float>(), samples, cache.audio.data() + cache.frame * samples);
        ++cache.frame;
        i.input_arena.reset();
        i.phase = Impl::Phase::Idle;
        break;
    }
    case Impl::Phase::Idle:
        if (!sampled.values.empty()) throw std::logic_error("unexpected idle sample result");
        break;
    default: throw std::logic_error("unexpected sampled completion");
    }
}

Completion ModelRunner::collect() {
    if (!finished() || !impl_->completion.executed) throw std::logic_error("execution is not complete");
    impl_->frame = {};
    impl_->scheduled = {};
    return std::exchange(impl_->completion, {});
}
PreparationBatch ModelRunner::collect_preparation() {
    if (!finished() || !preparing()) throw std::logic_error("preparation is not complete");
    return std::exchange(impl_->preparation_batch, {});
}
HostResults ModelRunner::collect_evaluation() {
    auto &i = *impl_;
    if (!finished() || i.evaluation.name.empty())
        throw std::logic_error("evaluation is not complete");
    i.evaluation = {};
    return std::move(i.evaluation_result);
}
std::vector<float> ModelRunner::collect_codes() {
    auto &i = *impl_;
    if (!finished() || !i.audio_decode)
        throw std::logic_error("code decode frame is not complete");
    auto cache = std::exchange(i.audio_decode, {});
    if (cache->frame == cache->codes.size() / i.info.audio.codebooks)
        return std::move(cache->audio);
    return {};
}
void ModelRunner::discard_control() noexcept {
    auto &i = *impl_;
    i.runtime.memory.complete_transfers();
    if (!preparing() && !has_control()) return;
    i.clear_control();
    i.input_arena.reset();
    i.preparation_batch = {};
    i.phase = Impl::Phase::Idle;
}

ForwardOutputs ModelRunner::preparation_forward() {
    auto &i = *impl_;
    auto &tasks = i.preparation_batch.tasks;
    while (i.preparation_first < tasks.size()) {
        const size_t first = i.preparation_first;
        if (tasks[first].error) { ++i.preparation_first; continue; }
        const auto &prototype = tasks[first].program;
        size_t end = first + 1;
        std::vector<const InputBatch *> sources{&prototype.input};
        while (end < tasks.size() && !tasks[end].error) {
            const auto &program = tasks[end].program;
            const auto &parameters = program.input.parameters;
            const auto &expected = prototype.input.parameters;
            if (program.name != prototype.name || program.outputs != prototype.outputs ||
                program.input.fixed_position != prototype.input.fixed_position ||
                parameters.size() != expected.size() || !std::equal(parameters.begin(), parameters.end(),
                    expected.begin(), [](const auto &a, const auto &b) { return a.name == b.name && a.value == b.value; })) break;
            sources.push_back(&program.input);
            ++end;
        }
        try {
            auto packed = i.pack(prototype.name, prototype.outputs, {sources.data(), sources.size()});
            i.preparation_end = first + packed.count;
            i.input_arena = std::make_unique<DeviceArena>(i.runtime.memory, packed.workspace_bytes);
            Tensor::Scope scope(*i.input_arena);
            const auto &input = packed.description.input;
            ForwardContext context(input.limits.batch, input.limits.time, input.limits.rows);
            // Output names must outlive the asynchronous readback. The task's
            // program remains owned until the preparation completion is consumed.
            auto outputs = output_views(prototype.outputs);
            run_forward(i.network, packed.description.name, input, {}, {outputs.data(), outputs.size()}, context);
            return {std::move(outputs)};
        } catch (const DeviceError &) { throw; }
        catch (const RequestOverloaded &) { end = first + 1; tasks[first].error = std::current_exception(); }
        catch (...) {
            auto error = std::current_exception();
            for (size_t index = first; index < end; ++index) tasks[index].error = error;
        }
        i.input_arena.reset();
        i.preparation_first = end;
    }
    i.phase = Impl::Phase::Idle;
    return {};
}

void ModelRunner::commit_preparation(const SampledOutputs &sampled) {
    auto &i = *impl_;
    auto &tasks = i.preparation_batch.tasks;
    const size_t first = i.preparation_first, end = i.preparation_end;
    TensorInputs outputs(sampled.values.data(), sampled.values.size());
    const auto &prototype = tasks[first].program;
    std::vector<int> offsets(prototype.outputs.size());
    for (size_t field = 0; field < offsets.size(); ++field) {
        int64_t rows = 0;
        for (size_t index = first; index < end; ++index) rows += tasks[index].outputs[field].rows;
        if (rows != outputs[prototype.outputs[field]].rows)
            throw std::logic_error("preparation output does not match its row partitions");
    }
    for (size_t index = first; index < end; ++index) {
        for (size_t field = 0; field < offsets.size(); ++field) {
            const auto &partition = tasks[index].outputs[field];
            auto source = outputs[partition.name].slice(offsets[field], partition.rows);
            offsets[field] += partition.rows;
            tasks[index].results.emplace(partition.name, Tensor(source));
        }
    }
    i.input_arena.reset();
    i.preparation_first = end;
}

size_t ModelRunner::max_step_samples() const {
    return size_t(impl_->definition.max_audio_frames) * impl_->info.audio.samples_per_frame +
           size_t(impl_->info.audio.sample_rate) * 10;
}
ResourceRequirements ModelRunner::requirements(const RequestPlan &plan) const {
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
    return {std::move(pages), workspace};
}
bool ModelRunner::release_if_idle(uint64_t id) {
    if (std::any_of(impl_->states.begin(), impl_->states.end(),
        [&](const auto &entry) { return entry.second.request == id; })) return false;
    impl_->release_request(id);
    return true;
}
std::shared_ptr<ModelRunner::Cache> ModelRunner::prepare(RequestPlan &plan, const ExecutionRow &row) {
    auto &i = *impl_;
    if (!row.cache || !row.cache->belongs_to(i.runtime.kv) || !row.slot || &row.slot->pool != &i.runtime.slots)
        throw std::invalid_argument("batch has no reserved cache resources");
    auto state = std::make_shared<Cache>(i.runtime, row.cache,
        row.slot, std::move(plan.inputs.at(row.input)), plan.options);
    i.start_segment(*state);
    return state;
}
ForwardOutputs ModelRunner::prefill_forward(const std::vector<Cache *> &requests) {
    Range trace("tts.runner.prefill");
    auto &implementation = *impl_;
    std::vector<Impl::Cache *> rows;
    std::vector<const InputBatch *> sources;
    for (auto *request : requests) {
        auto &state = *request;
        if (state.is_prefilled || state.finished) continue;
        rows.push_back(&state);
        sources.push_back(&state.current().input);
    }
    if (rows.empty()) return {};
    auto program = implementation.pack(implementation.definition.main_program,
        {"hidden_states", "logits", "done"}, {sources.data(), sources.size()},
        size_t(implementation.runtime.kv.block_stride()) * sizeof(int));
    rows.resize(program.count);
    auto &packed = program.description.input;
    std::vector<int> blocks;
    blocks.reserve(size_t(packed.sequence.shape.batch) * implementation.runtime.kv.block_stride());
    for (const auto *state : rows) {
        const auto &table = state->kv->block_table();
        blocks.insert(blocks.end(), table.begin(), table.end());
    }
    const size_t bytes = program.workspace_bytes + Tensor::aligned(blocks.size() * sizeof(int));
    implementation.input_arena = std::make_unique<DeviceArena>(implementation.runtime.memory, bytes);
    auto block_values = std::make_shared<Tensor>(Tensor::zeros({int(blocks.size()), 1}, Type::I32));
    std::copy(blocks.begin(), blocks.end(), block_values->view().ptr<int>());
    packed.uploads.push_back({"cache_blocks", block_values->view()});
    packed.owners.push_back(block_values);
    packed.batch_bindings.push_back({"cache_blocks", BatchField::CacheTable});
    implementation.map(rows);
    Tensor::Scope scope(*implementation.input_arena);
    ForwardContext context(packed.limits.batch, packed.limits.time, packed.limits.rows);
    context.batch.capacity = implementation.runtime.context;
    context.batch.cache = implementation.runtime.kv.view();
    auto arguments = implementation.bindings(int(rows.size()), packed.sequence.shape.batch);
    auto outputs = output_views(program.description.outputs);
    run_forward(implementation.network, program.description.name, packed, {arguments.data(), arguments.size()},
                {outputs.data(), outputs.size()}, context);
    implementation.input_rows = std::move(rows);
    return {{{"done", implementation.done.view(int(implementation.input_rows.size()), 1, Type::I32)}}};
}
ForwardOutputs ModelRunner::decode_forward() {
    Range trace("tts.runner.decode");
    auto &implementation = *impl_;
    auto &rows = implementation.step_rows;
    auto &decode = implementation.decode_rows;
    rows.clear(); decode.clear();
    implementation.audio_sinks.clear();
    implementation.active_audio.clear();
    for (size_t index = 0; index < implementation.scheduled.rows.size(); ++index) {
        const auto &row = implementation.scheduled.rows[index];
        const auto &out = implementation.completion.rows[index];
        const auto found = implementation.states.find(row.id);
        if (out.done || found == implementation.states.end()) continue;
        auto &cache = *found->second.value;
        if (cache.finished || !cache.is_prefilled) continue;
        rows.push_back(&cache);
        implementation.audio_sinks.push_back(&implementation.completion.rows[index].pcm);
        if (!cache.semantic_done && !implementation.ready(cache)) decode.push_back(&cache);
    }
    implementation.completion.decoded_rows = int(rows.size());
    if (implementation.frame.slot >= 0) {
        implementation.phase = Impl::Phase::Frame;
        return frame_outputs();
    }
    if (!rows.empty() && std::all_of(rows.begin(), rows.end(),
        [&](const auto *row) { return implementation.framewise(*row); })) {
        implementation.frame = implementation.launch_frame(implementation.scheduled);
        if (implementation.frame.slot >= 0) {
            implementation.phase = Impl::Phase::Frame;
            return frame_outputs();
        }
    }
    if (!decode.empty()) implementation.launch_decode(decode);
    if (decode.empty()) { implementation.phase = Impl::Phase::Audio; return forward(); }
    return {{{"done", implementation.done.view(int(decode.size()), 1, Type::I32)}}};
}
ForwardOutputs ModelRunner::frame_outputs() const {
    const auto &i = *impl_;
    const auto &storage = i.frame_storage.at(i.frame.slot);
    const int rows = int(i.frame.rows.size());
    return {{{"done", storage.view(rows, 1, Type::I32)},
             {"pcm", storage.view(rows, i.info.audio.samples_per_frame, Type::I16, Tensor::aligned(i.done.size()))}}};
}
void ModelRunner::commit_frame(const SampledOutputs &sampled) {
    auto &i = *impl_;
    TensorInputs values(sampled.values.data(), sampled.values.size());
    const auto done = values["done"], pcm = values["pcm"];
    const int samples = i.info.audio.samples_per_frame;
    if (done.numel() != i.frame.rows.size() || pcm.numel() != i.frame.rows.size() * samples)
        throw std::logic_error("frame completion does not match submitted rows");
    for (size_t index = 0; index < i.frame.rows.size(); ++index) {
        const auto &row = i.frame.rows[index];
        auto &cache = *row.cache;
        // EOS of the preceding frame can retire this cache while one final
        // frame is still executing. The packet owns it; discard that result.
        if (cache.finished || !i.states.count(row.id)) continue;
        auto out = std::find_if(i.completion.rows.begin(), i.completion.rows.begin() + i.completion.valid_rows,
            [&](const CompletedRow &entry) { return entry.id == row.id; });
        if (out == i.completion.rows.begin() + i.completion.valid_rows) continue;
        if (row.decoded) {
            if (done.ptr<int>()[index]) { cache.semantic_done = true; continue; }
            if (++cache.generated == cache.current().max_tokens) {
                cache.semantic_done = true;
                cache.truncated = true;
            }
        }
        i.append_pcm(cache, out->pcm, pcm.ptr<int16_t>() + index * samples, samples);
        ++cache.rendered; ++cache.audio_rendered; ++cache.windows;
    }
    i.phase = Impl::Phase::Finish;
}
ForwardOutputs ModelRunner::audio_forward() {
    auto &implementation = *impl_;
    auto &runtime = implementation.runtime;
    auto &rows = implementation.step_rows;
    auto ready = [&](const Impl::Cache &state) { return implementation.ready(state); };
    const auto &description = implementation.program(implementation.definition.audio_program, ForwardMode::Prefill);
    auto &audio = implementation.active_audio;
    auto &members = implementation.audio_members;
    audio.clear(); members.clear();
    for (size_t row = 0; row < rows.size(); ++row)
        if (!rows[row]->finished && (ready(*rows[row]) || implementation.audio_pending(*rows[row]))) {
            members.push_back(row);
            audio.push_back(rows[row]);
        }
    if (!audio.empty()) {
        implementation.audio_execution = implementation.audio_description(audio);
        implementation.launch_audio(audio, implementation.audio_execution);
        const bool indexed = std::find(description.outputs.begin(), description.outputs.end(), "audio_layout") != description.outputs.end();
        const int stride = implementation.audio_stride;
        const size_t total = indexed ? implementation.audio_output.numel() : size_t(stride) * audio.size();
        if ((!indexed && stride < 1) || total > implementation.host_audio.size())
            throw std::logic_error("model audio exceeds its declared output capacity");
        implementation.audio_total = total;
        return {{{"pcm", TensorView(implementation.audio_output.data, int(total), 1,
                                    Type::I16, 0, runtime.memory.device)}}};
    }
    implementation.phase = Impl::Phase::Finish;
    return forward();
}
void ModelRunner::commit_audio(const SampledOutputs &sampled) {
    auto &implementation = *impl_;
    auto &audio = implementation.active_audio;
    auto &members = implementation.audio_members;
    const auto &execution = implementation.audio_execution;
    const auto &description = implementation.program(implementation.definition.audio_program, ForwardMode::Prefill);
    const bool indexed = std::find(description.outputs.begin(), description.outputs.end(), "audio_layout") != description.outputs.end();
    const int stride = implementation.audio_stride;
    const size_t total = implementation.audio_total;
    auto available = [&](const Impl::Cache &state) { return implementation.available(state); };
    TensorInputs values(sampled.values.data(), sampled.values.size());
    auto *host_pcm = values["pcm"].ptr<int16_t>();
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
        auto *current = host_pcm + offset + begin;
        auto *output = current - state.pending.size();
        for (size_t sample = 0; sample < state.pending.size(); ++sample) {
            const float alpha = state.pending.size() > 1 ? float(sample) / (state.pending.size() - 1) : 1;
            output[sample] = int16_t(std::lround(state.pending[sample] * (1 - alpha) + output[sample] * alpha));
        }
        const bool final = state.semantic_done && state.rendered == available(state);
        const int send = final ? length : length - state.input.windows.fade * samples_per_frame;
        const size_t emitted = state.pending.size() + send;
        state.pending.assign(current + send, current + length);
        implementation.append_pcm(state, *implementation.audio_sinks[members[row]], output, emitted);
        state.audio_rendered += int(emitted / samples_per_frame);
        if (state.input.windows.mode == SynthesisMode::Hybrid) {
            const double now = std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
            state.schedule.observe(accepted, emitted, implementation.info.audio.sample_rate, now);
        }
    }
    implementation.active_audio.clear();
    implementation.phase = Impl::Phase::Finish;
}
void ModelRunner::finish_step() {
    auto &implementation = *impl_;
    auto &rows = implementation.step_rows;
    auto available = [&](const Impl::Cache &state) { return implementation.available(state); };
    for (size_t row = 0; row < rows.size(); ++row) {
        auto *state = rows[row];
        if (!state->finished && state->semantic_done && state->rendered >= available(*state) &&
            !implementation.audio_pending(*state)) {
            implementation.append_pcm(*state, *implementation.audio_sinks[row], state->pending.data(), state->pending.size());
            state->pending.clear();
            if (state->segment + 1 < state->input.segments.size() && state->emitted_frames < state->options.max_frames) {
                int pause = std::min(state->input.pause_frames, state->options.max_frames - state->emitted_frames);
                std::fill(implementation.host_audio.begin(), implementation.host_audio.end(), int16_t{});
                while (pause && !state->finished) {
                    const int count = std::min(pause, implementation.definition.max_audio_frames);
                    implementation.append_pcm(*state, *implementation.audio_sinks[row], implementation.host_audio.data(),
                        size_t(count) * implementation.info.audio.samples_per_frame);
                    pause -= count;
                }
                if (!state->finished && state->emitted_frames < state->options.max_frames) {
                    ++state->segment;
                    implementation.start_segment(*state);
                } else state->finished = true;
            } else state->finished = true;
        }
        if (state->emitted_frames >= state->options.max_frames) state->finished = true;
    }
}
ForwardOutputs ModelRunner::forward_reference(const ReferencePlan &plan) {
    auto &i = *impl_;
    if (i.has_work())
        throw std::logic_error("runner already has scheduled work");
    i.evaluation = {plan.program, plan.input, plan.outputs};
    i.evaluation_result.clear();
    i.phase = Impl::Phase::Evaluation;
    return forward();
}
ForwardOutputs ModelRunner::forward_encode(const std::vector<float> &mono) {
    auto &i = *impl_;
    if (mono.empty() || mono.size() > i.info.max_reference_samples)
        throw std::invalid_argument("reference audio is empty or exceeds its declared capacity");
    ReferencePlan plan;
    plan.program = i.definition.reference_program;
    plan.input.limits = {1, int(mono.size()), int(mono.size())};
    plan.input.sequence.shape = {1, int(mono.size()), int(mono.size())};
    plan.input.uploads.push_back({"waveform", TensorView(const_cast<float *>(mono.data()), int(mono.size()), 1, Type::F32)});
    plan.outputs = {"codes"};
    return forward_reference(plan);
}
std::shared_ptr<ModelRunner::AudioDecodeCache> ModelRunner::prepare_codes(const std::vector<int> &codes) {
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
    if (implementation.has_work())
        throw std::logic_error("runner already has scheduled work");
    auto result = std::make_shared<AudioDecodeCache>();
    // Retain ownership before submitting initialization, including error cleanup.
    implementation.audio_decode = result;
    result->codes = codes;
    result->workspace_bytes = network_bytes;
    result->arena = std::make_unique<DeviceArena>(implementation.runtime.memory, history_bytes);
    Tensor::Scope allocation_scope(*result->arena);
    result->history = std::make_unique<SequenceState>(specification);
    auto &history = *result->history;
    history.reset(implementation.runtime.memory.stream);
    result->token = Buffer(size_t(format.codebooks) * sizeof(int));
    result->pointer = Buffer(sizeof(int *));
    result->cache = Buffer(sizeof(CacheSequence));
    result->histories = Buffer(specification.history_bytes.size() * sizeof(ConvHistory));
    const int *frame_pointer = result->token.ptr<int>();
    CacheSequence sequence{history.keys.data(), history.values.data(), history.layer_bytes, nullptr, history.position.ptr<int>()};
    std::vector<ConvHistory> descriptors;
    for (auto &buffer : history.history) descriptors.push_back({buffer.ptr<float>(), history.position.ptr<int>()});
    result->pointer.copy_({&frame_pointer, 1, 1, Type::I64}, true);
    result->cache.copy_({&sequence, 1, int(sizeof(sequence)), Type::I8}, true);
    result->histories.copy_({descriptors.data(), 1, int(descriptors.size() * sizeof(ConvHistory)), Type::I8}, true);
    result->audio.resize(codes.size() / format.codebooks * format.samples_per_frame);
    implementation.audio_decode.reset();
    return result;
}
ForwardOutputs ModelRunner::forward_codes(std::shared_ptr<AudioDecodeCache> cache) {
    auto &i = *impl_;
    if (i.has_work()) throw std::logic_error("runner already has scheduled work");
    if (!cache || cache->frame >= cache->codes.size() / i.info.audio.codebooks)
        throw std::logic_error("code decode cache is complete");
    i.audio_decode = std::move(cache);
    i.input_arena = std::make_unique<DeviceArena>(i.runtime.memory, i.audio_decode->workspace_bytes);
    i.phase = Impl::Phase::Codes;
    return forward();
}

ForwardOutputs ModelRunner::decode_codes_forward() {
    auto &i = *impl_;
    auto &cache = *i.audio_decode;
    const auto &format = i.info.audio;
    const auto &description = i.program(i.definition.audio_program, ForwardMode::Prefill);
    auto input = description.input;
    input.limits = {1, 1, 1};
    input.sequence = {{}, {1, 1, 1}};
    input.parameters = {{"history_stride", 1}};
    cache.token.copy_({cache.codes.data() + cache.frame * format.codebooks,
                          1, format.codebooks, Type::I32}, true);
    Tensor::Scope allocation_scope(*i.input_arena);
    Tensor::Scope scratch;
    ForwardContext context(1, 1, 1);
    context.batch.cache.layout.sequences = cache.cache.ptr<CacheSequence>();
    const NamedTensor bindings[] = {{"code_rows", cache.pointer.view(1, 1, Type::I64)},
        {"histories", cache.histories.view(1, int(cache.histories.size()), Type::I8)}};
    auto outputs = output_views(description.outputs);
    run_forward(i.network, description.name, input, {bindings, std::size(bindings)},
                {outputs.data(), outputs.size()}, context);
    TensorInputs values(outputs.data(), outputs.size());
    return {{{"waveform", values["waveform"]}}};
}
} // namespace tts
