#include "ops/sampling.h"
#include "ops/audio.h"
#include "models/breeze/model_runner.h"
#include "models/breeze/generation_state.h"
#include "runtime/execution_resources.h"
#include "runtime/kv_cache.h"
#include "runtime/request_admission.h"
#include "models/breeze/prepared_batch.h"
#include "models/breeze/request_planner.h"
#include "models/breeze/prefill.h"
#include "support/tensor_debug.h"
#include "support/log.h"
#include <chrono>
#include <array>
#include <cmath>
#include <fstream>
#include <filesystem>
namespace tts::breeze {
using Clock = std::chrono::steady_clock;
static double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
namespace {
// This model supports unconditional, instruction and reference guidance branches.
struct RequestBuffers {
    SequenceState cs;
    Buffer codes, done, positions, rng, seen, table;
    RequestBuffers(const Model &m, int blocks, BufferAllocator *storage)
        : cs(m.codec.history_spec(), storage), codes(m.config.codebooks * sizeof(int), storage),
          done(sizeof(int), storage), positions(max_guidance_branches * sizeof(int), storage),
          rng(sizeof(uint64_t), storage), seen(m.config.codec.codebook_size * sizeof(int), storage),
          table(size_t(max_guidance_branches) * blocks * sizeof(int), storage) {}
};
} // namespace
struct Runner::Slot {
    DeviceArena storage;
    std::unique_ptr<RequestBuffers> buffers;
    Slot(Runner &r) : storage(r.gpu.memory) {
        WorkspacePlan plan;
        RequestBuffers layout(r.model, r.kv_->block_stride(), &plan);
        storage.reserve(plan.size);
        buffers = std::make_unique<RequestBuffers>(r.model, r.kv_->block_stride(), &storage);
    }
};
struct Runner::PrefillWork {
    std::vector<uint64_t> ids;
    PreparedBatch prepared;
    std::vector<int> blocks;
    DeviceArena scratch;
    ArenaLease lease;
    Buffer table;
    Prefill state;
    static size_t workspace_bytes(Runner &r, const PreparedBatch &p) {
        return prefill_workspace_bytes(r.model, p, r.kv_->view()) +
               BufferAllocator::aligned(size_t(p.lanes) * r.kv_->block_stride() * sizeof(int));
    }
    PrefillWork(Runner &r, std::vector<uint64_t> identities, PreparedBatch p,
                std::vector<int> blocks);
};
struct Runner::State final : RequestState {
    Runner &owner;
    std::unique_ptr<RequestAdmissionPool::Lease::RowLease> row_credit;
    std::unique_ptr<RequestSlots::Lease> slot;
    RequestBuffers *state;
    const GenerationOptions opt;
    const uint64_t id;
    PreparedBatch prepared;
    std::unique_ptr<PagedKVCache::Lease> kv_lease;
    std::vector<int> ended;
    std::ofstream token_out;
    Clock::time_point start = Clock::now();
    Stats metrics;
    std::exception_ptr failure;
    int frame = 0, delivered = 0;
    bool finished = false, cancelled = false, is_prefilled = false;
    State(Runner &r, std::unique_ptr<RequestAdmissionPool::Lease::RowLease> credit,
          std::unique_ptr<RequestSlots::Lease> lease, GenerationOptions options, uint64_t identity)
        : owner(r), row_credit(std::move(credit)), slot(std::move(lease)), state(r.slots_[slot->index]->buffers.get()),
          opt(options), id(identity) {}
    bool done() const override { return finished; }
    bool prefilled() const override { return is_prefilled; }
    Stats stats() const override { return metrics; }
    std::exception_ptr error() const override { return failure; }
    ~State() override {
        if (owner.prefill_ && std::find(owner.prefill_->ids.begin(), owner.prefill_->ids.end(),
                                        id) != owner.prefill_->ids.end())
            owner.prefill_.reset();
    }
};
// A single dense compute workspace, with borrowed pointers to request-owned state.
// Membership updates upload only small descriptors; KV, RNG and codec histories never move.
struct Runner::Batch {
    struct Metadata {
        Buffer sampling, code_rows, lane_rows, caches, codec_codes, codec_caches, histories;
        Metadata() = default;
        Metadata(int rows, int lanes, size_t convolutions, BufferAllocator *storage)
            : sampling(rows * sizeof(SamplingRow), storage),
              code_rows(rows * sizeof(int *), storage), lane_rows(lanes * sizeof(int), storage),
              caches(lanes * sizeof(CacheSequence), storage),
              codec_codes(rows * sizeof(int *), storage),
              codec_caches(rows * sizeof(CacheSequence), storage),
              histories(convolutions * rows * sizeof(ConvHistory), storage) {}
    };
    struct Row {
        State *request;
        int lane;
        uint64_t id;
        bool operator==(const Row &other) const {
            return id == other.id && lane == other.lane;
        }
    };
    ExecutionResources storage;
    std::unique_ptr<ArenaLease> lease;
    std::unique_ptr<GenerationState> state;
    Metadata metadata;
    std::vector<Row> decode_rows, audio_rows;
    std::vector<int16_t> pcm;
    std::vector<int> ended;
    int lanes = 0, decode_capacity = 0, audio_capacity = 0, capacity, lane_capacity;

    static size_t workspace_bytes(Runner &r, int rows, int lanes) {
        WorkspacePlan plan;
        GenerationState planned(r.model, r.context, rows, lanes, lanes, &plan, r.kv_->view());
        Metadata metadata(rows, lanes, planned.cs.conv.size(), &plan);
        return plan.size;
    }
    Batch(Runner &r, int rows, int lane_count)
        : storage(r.gpu.memory), capacity(rows), lane_capacity(lane_count) {
        storage.prepare(workspace_bytes(r, rows, lane_count));
        lease.reset(new ArenaLease(storage.lease_workspace()));
        state =
            std::make_unique<GenerationState>(r.model, r.context, capacity, lane_capacity,
                                              lane_capacity, &storage.workspace(), r.kv_->view());
        metadata = Metadata(capacity, lane_capacity, state->cs.conv.size(), &storage.workspace());
        pcm.resize(size_t(capacity) * r.model.config.samples_per_frame);
        ended.resize(capacity);
    }
    void map_decode(Runner &r, const std::vector<Row> &rows, int padded_rows = 0,
                    int padded_lanes = 0) {
        if (!padded_rows)
            padded_rows = int(rows.size());
        if (!padded_lanes)
            for (const auto &row : rows)
                padded_lanes += row.request->prepared.branches;
        if (rows == decode_rows && padded_rows == decode_capacity && padded_lanes == lanes)
            return;
        std::vector<SamplingRow> samples;
        std::vector<const int *> codes;
        std::vector<int> owners;
        std::vector<CacheSequence> sequences;
        for (const auto &row : rows) {
            auto &request = *row.request;
            auto &s = *request.state;
            const auto &opt = request.opt;
            auto kv = request.kv_lease->view();
            SamplingRow sample;
            sample.codes = s.codes.ptr<int>() + row.lane * r.model.config.codebooks;
            sample.done = s.done.ptr<int>() + row.lane;
            sample.seen = s.seen.ptr<int>() + row.lane * r.model.config.codec.codebook_size;
            sample.rng = s.rng.ptr<uint64_t>() + row.lane;
            sample.branches = request.prepared.branches;
            sample.cfg = opt.cfg_scale;
            sample.cfg_ref = opt.cfg_scale_ref;
            sample.cfg_ins = opt.cfg_scale_ins;
            sample.penalty = opt.repetition_penalty;
            sample.backbone = {opt.do_sample ? opt.temperature : 0.f, opt.top_p, opt.top_k};
            sample.depth = {opt.depth_do_sample ? opt.depth_temperature : 0.f, opt.depth_top_p,
                            opt.depth_top_k};
            for (int branch = 0; branch < sample.branches; ++branch) {
                int lane = branch * request.prepared.batch + row.lane;
                sample.logits[branch] = int(sequences.size());
                owners.push_back(int(samples.size()));
                sequences.push_back({kv.keys.data, kv.values.data,
                                     size_t(kv.rows_per_layer) * kv.keys.cols * bytes(kv.keys.type),
                                     kv.layout.block_table + lane * kv.layout.block_stride,
                                     s.positions.ptr<int>() + lane});
            }
            codes.push_back(sample.codes);
            samples.push_back(sample);
        }
        samples.resize(padded_rows);
        codes.resize(padded_rows, nullptr);
        owners.resize(padded_lanes, -1);
        sequences.resize(padded_lanes);
        decode_capacity = padded_rows;
        r.gpu.upload(metadata.sampling.data(), samples.data(),
                     samples.size() * sizeof(SamplingRow));
        r.gpu.upload(metadata.code_rows.data(), codes.data(), codes.size() * sizeof(int *));
        r.gpu.upload(metadata.lane_rows.data(), owners.data(), owners.size() * sizeof(int));
        r.gpu.upload(metadata.caches.data(), sequences.data(),
                     sequences.size() * sizeof(CacheSequence));
        lanes = int(sequences.size());
        decode_rows = rows;
    }
    void map_codec(Runner &r, const std::vector<Row> &rows, int padded_rows) {
        if (rows == audio_rows && audio_capacity == padded_rows)
            return;
        std::vector<const int *> codes;
        std::vector<CacheSequence> sequences;
        std::vector<ConvHistory> history(state->cs.conv.size() * capacity);
        for (size_t i = 0; i < rows.size(); ++i) {
            const auto &row = rows[i];
            auto &s = *row.request->state;
            auto &cs = s.cs;
            size_t layer_bytes = cs.keys.size() / r.model.codec.transformer.config.layers;
            sequences.push_back(
                {cs.keys.data(), cs.values.data(), layer_bytes, nullptr, cs.position.ptr<int>()});
            codes.push_back(s.codes.ptr<int>() + row.lane * r.model.config.codebooks);
            for (size_t op = 0; op < cs.history.size(); ++op)
                history[op * capacity + i] = {cs.history[op].ptr<float>(),
                                              cs.position.ptr<int>() + row.lane};
        }
        codes.resize(padded_rows, nullptr);
        sequences.resize(padded_rows);
        audio_capacity = padded_rows;
        r.gpu.upload(metadata.codec_codes.data(), codes.data(), codes.size() * sizeof(int *));
        r.gpu.upload(metadata.codec_caches.data(), sequences.data(),
                     sequences.size() * sizeof(CacheSequence));
        r.gpu.upload(metadata.histories.data(), history.data(),
                     history.size() * sizeof(ConvHistory));
        audio_rows = rows;
    }
    void read_done(Backend &gpu, const std::vector<Row> &rows) {
        gpu.download(ended.data(), state->done.data(), rows.size() * sizeof(int));
        for (size_t i = 0; i < rows.size(); ++i)
            rows[i].request->ended[rows[i].lane] = ended[i];
    }
};
Runner::PrefillWork::PrefillWork(Runner &r, std::vector<uint64_t> identities,
                                      PreparedBatch p, std::vector<int> mapping)
    : ids(std::move(identities)), prepared(std::move(p)), blocks(std::move(mapping)),
      scratch(r.gpu.memory, workspace_bytes(r, prepared)), lease(scratch),
      table(blocks.size() * sizeof(int), &scratch),
      state(r.model, scratch, prepared, r.kv_->view(table.ptr<int>()),
            r.batch_->state->last.view(prepared.lanes, r.model.backbone.config.hidden)) {
    r.gpu.upload(table.data(), blocks.data(), table.size());
}
Runner::Runner(const Model &m, Backend &g, int batch, int ctx, int page,
                         size_t runtime_memory_bytes)
    : model(m), gpu(g), max_batch(batch), context(ctx), graph_shapes_(batch),
      kv_(std::make_unique<PagedKVCache>(
          g.memory, m.backbone.config.layers, m.backbone.config.kvheads * m.backbone.config.dim,
          m.backbone.config.type, batch * max_guidance_branches, ctx, page)),
      admission_(std::make_unique<RequestAdmissionPool>(g.memory, *kv_, batch)), slot_pool_(batch) {
    ExecutionScope execution(gpu);
    slots_.reserve(max_batch);
    gpu.reserve_attention_workspace(size_t(batch) * max_guidance_branches * ctx *
                                    std::max(m.backbone.config.heads, 2 * m.text.config.heads));
    batch_ = std::make_unique<Batch>(*this, max_batch, max_batch * max_guidance_branches);
    for (int i = 0; i < max_batch; ++i)
        slots_.push_back(std::make_unique<Slot>(*this));
    // Capacity classes cover arbitrary prefill lengths without padding or an
    // exact-shape plan table. Configuration and library descriptors are init-only.
    auto prepare = [&](const char *name, const Transformer &transformer, int rows) {
        auto start = Clock::now();
        TTS_LOG(TTS_LOG_INFO, "engine.init", "planning start transformer={} max_rows={}", name,
                rows);
        transformer.prepare_range(rows);
        TTS_LOG(TTS_LOG_INFO, "engine.init", "planning complete transformer={} ms={:.0f}", name,
                milliseconds(start));
    };
    prepare("text", model.text, max_batch * max_guidance_branches * context);
    prepare("backbone", model.backbone, max_batch * max_guidance_branches * context);
    prepare("depth", model.depth, 2 * max_guidance_branches * max_batch);
    const std::array<std::pair<const Transformer *, int>, 3> transformers{{
        {&model.text, max_batch * max_guidance_branches * context},
        {&model.backbone, max_batch * max_guidance_branches * context},
        {&model.depth, 2 * max_guidance_branches * max_batch}}};
    size_t input_bytes = 0, output_bytes = 0;
    auto visit = [&](auto use) {
        for (auto entry : transformers)
            for (const auto &layer : entry.first->layers)
                for (auto weight : {layer.qkv, layer.out, layer.gateup, layer.down})
                    if (weight.data) use(weight, entry.second);
        use(model.text_proj, max_batch * max_guidance_branches * context);
    };
    visit([&](TensorView weight, int rows) {
        input_bytes = std::max(input_bytes, size_t(rows) * weight.cols * bytes(weight.type));
        output_bytes = std::max(output_bytes, size_t(rows) * weight.rows * bytes(weight.type));
    });
    Buffer planning_input(input_bytes, &gpu.memory), planning_output(output_bytes, &gpu.memory);
    visit([&](TensorView weight, int rows) {
        gpu.prepare_linear(planning_input.view(rows, weight.cols, weight.type), weight,
                           planning_output.view(rows, weight.rows, weight.type));
    });
    auto graph_start = Clock::now();
    TTS_LOG(TTS_LOG_INFO, "engine.init", "CUDA graph capture start");
    initialize_graphs();
    TTS_LOG(TTS_LOG_INFO, "engine.init", "CUDA graph capture complete ms={:.0f}",
            milliseconds(graph_start));

    auto encoder_planning = model.encoder.prepare(gpu, model.config.max_reference_frames * model.config.samples_per_frame);
    gpu.seal_plans();
    encoder_planning.reset();
    planning_input = Buffer(); planning_output = Buffer();
    // Default backing covers full KV credits and the largest single admission's
    // prefill. Reference encoding shares this backing and is checked by its actual
    // workspace size. An explicit budget permits a different concurrency/audio mix.
    const size_t eager_bytes =
        prefill_workspace_bound(model, context, max_guidance_branches, kv_->view()) +
        BufferAllocator::aligned(size_t(max_guidance_branches) * kv_->block_stride() * sizeof(int));
    gpu.memory.trim();
    const size_t available = gpu.memory.available_bytes();
    const size_t budget = runtime_memory_bytes ? runtime_memory_bytes
                                               : std::min(eager_bytes + kv_->max_storage_bytes(),
                                                          available - available / 10);
    if (!budget || budget > available)
        throw DeviceOutOfMemory("runtime memory reservation exceeds available GPU memory");
    WorkspacePlan transfers, reference;
    transfers.allocate(prefill_transfer_bound(model, context, max_batch * max_guidance_branches));
    for (const auto &slot : slots_) {
        const auto &s = *slot->buffers;
        for (auto *b : {&s.rng, &s.positions, &s.table, &s.table}) transfers.allocate(b->size());
        // Each new physical-page chunk uploads its K/V address tables once.
        for (int i = 0; i < 2; ++i)
            transfers.allocate(size_t(max_guidance_branches) * kv_->block_stride() * sizeof(void *));
    }
    const auto &meta = batch_->metadata;
    for (auto *b : {&meta.sampling, &meta.code_rows, &meta.lane_rows, &meta.caches,
                    &meta.codec_codes, &meta.codec_caches, &meta.histories})
        transfers.allocate(b->size());
    transfers.allocate(size_t(max_batch) * sizeof(int));
    transfers.allocate(size_t(max_batch) * model.config.samples_per_frame * sizeof(int16_t));
    reference.allocate(size_t(model.config.max_reference_frames) *
                       model.config.samples_per_frame * sizeof(float));
    reference.allocate(size_t(model.config.max_reference_frames) *
                       model.config.codebooks * sizeof(int));
    reference.allocate(size_t(model.config.samples_per_frame) * sizeof(float));
    gpu.memory.reserve_transfers(std::max(transfers.size, reference.size));
    gpu.memory.reserve_runtime(budget);
}
void Runner::initialize_graphs() {
    auto &batch = *batch_;
    auto &graphs = batch.storage.graphs();
    for (int rows : graph_shapes_.rows()) {
        for (int branches = 1; branches <= max_guidance_branches; ++branches)
            graphs.register_shape("decode", {rows, rows * branches});
        graphs.register_shape("codec", {rows});
    }
    graphs.capture("decode", [&](const auto &shape) {
        DecodeBatch view{shape[1], batch.metadata.sampling.ptr<SamplingRow>(),
                         batch.metadata.code_rows.ptr<const int *>(),
                         batch.metadata.lane_rows.ptr<int>(),
                         batch.metadata.caches.ptr<CacheSequence>()};
        model.next_frame(*batch.state, shape[0], view);
    });
    graphs.capture("codec", [&](const auto &shape) {
        CodecBatch view{batch.metadata.codec_codes.ptr<const int *>(),
                        batch.metadata.codec_caches.ptr<CacheSequence>(),
                        batch.metadata.histories.ptr<ConvHistory>(), batch.capacity};
        model.codec.forward(batch.state->cs, nullptr, shape[0], &view);
    });
    batch.storage.seal();
}
Runner::~Runner() = default;
size_t Runner::max_step_samples() const {
    return size_t(model.config.samples_per_frame);
}
struct Runner::Admission final : RequestAdmission {
    Runner &owner;
    std::unique_ptr<RequestAdmissionPool::Lease> resources;
    Admission(Runner &runner, std::unique_ptr<RequestAdmissionPool::Lease> lease)
        : owner(runner), resources(std::move(lease)) {}
};
std::unique_ptr<Runner::RequestAdmission> Runner::admit(RequestPlan &base) try {
    auto &plan = dynamic_cast<Plan &>(base);
    CUDA(cudaSetDevice(gpu.device));
    ExecutionScope execution(gpu);
    Range admission("tts.batch.admission");
    std::vector<int> pages;
    size_t workspace = 0;
    for (const auto &input : plan.inputs) {
        pages.push_back(kv_->required_pages(input.capacities));
        workspace = std::max(workspace, PrefillWork::workspace_bytes(*this, input));
    }
    auto lease = admission_->admit(pages, workspace);
    if (!lease) return {};
    return std::make_unique<Admission>(*this, std::move(lease));
} catch (const DeviceOutOfMemory &error) {
    throw RequestOverloaded(error.what());
}
std::unique_ptr<Runner::RequestState> Runner::prepare(RequestPlan &base, size_t lane,
                                                      RequestAdmission &reservation) try {
    auto &plan = dynamic_cast<Plan &>(base);
    auto &admission = dynamic_cast<Admission &>(reservation);
    CUDA(cudaSetDevice(gpu.device));
    ExecutionScope execution(gpu);
    Range preparation("tts.batch.admission");
    if (&admission.owner != this)
        throw std::invalid_argument("request admission belongs to another runner");
    auto &input = plan.inputs.at(lane);
    auto row_credit = admission.resources->acquire_row();
    if (!row_credit) return {};
    auto prepared = std::move(input);
    auto options = plan.options;
    options.seed = request_seed(options.seed, lane);
    auto impl = std::make_unique<State>(*this, std::move(row_credit), slot_pool_.acquire(),
                                        options, ++request_sequence_);
    auto &s = *impl;
    s.prepared = std::move(prepared);
    s.ended.resize(s.prepared.batch);
    TensorDebugScope debug_scope(s.id);
    s.kv_lease = std::make_unique<PagedKVCache::Lease>(
        *kv_, s.prepared.lengths, s.prepared.capacities, s.state->table.ptr<int>());
    gpu.zero(s.state->codes);
    gpu.zero(s.state->done);
    gpu.zero(s.state->seen);
    s.state->cs.reset(gpu.stream);
    gpu.upload(s.state->rng.data(), &s.opt.seed, sizeof(uint64_t));
    gpu.upload(s.state->positions.data(), s.prepared.lengths.data(),
               s.prepared.lengths.size() * sizeof(int));
    if (const char *path = std::getenv("TTS_TOKENS_OUT")) {
        s.token_out.open(std::filesystem::u8path(debug_path(path, s.id)), std::ios::binary);
        if (!s.token_out)
            throw std::runtime_error("cannot create token debug output");
    }
    return impl;
} catch (const DeviceOutOfMemory &error) {
    // Admission failure does not alter another request's state or grow backing.
    throw RequestOverloaded(error.what());
}
size_t Runner::prefill_step(const std::vector<RequestState *> &requests) {
    CUDA(cudaSetDevice(gpu.device));
    ExecutionScope execution(gpu);
    if (!prefill_) {
        std::vector<const PreparedBatch *> inputs;
        std::vector<uint64_t> ids;
        std::vector<int> blocks;
        PreparedBatch packed;
        for (auto *request : requests) {
            auto &s = state(*request);
            if (s.is_prefilled || s.finished)
                continue;
            inputs.push_back(&s.prepared);
            auto candidate = pack_prefill(inputs);
            if (!gpu.memory.can_allocate({PrefillWork::workspace_bytes(*this, candidate)})) {
                inputs.pop_back();
                if (ids.empty())
                    throw std::logic_error("prefill admission workspace guarantee violated");
                break;
            }
            packed = std::move(candidate);
            ids.push_back(s.id);
            const auto &table = s.kv_lease->block_table();
            blocks.insert(blocks.end(), table.begin(), table.end());
        }
        if (ids.empty())
            return 0;
        prefill_ = std::make_unique<PrefillWork>(*this, std::move(ids), std::move(packed),
                                                 std::move(blocks));
    }
    std::vector<Batch::Row> rows;
    for (auto id : prefill_->ids) {
        auto found = std::find_if(requests.begin(), requests.end(),
                                  [&](auto *request) { return state(*request).id == id; });
        if (found == requests.end())
            throw std::logic_error("active prefill input missing from execution batch");
        rows.push_back({&state(**found), 0, id});
    }
    try {
        TensorDebugScope debug_scope(rows.front().id);
        if (!prefill_->state.step())
            return rows.size();
        prefill_.reset();
        batch_->map_decode(*this, rows);
        DecodeBatch mapped{batch_->lanes, batch_->metadata.sampling.ptr<SamplingRow>(),
                           batch_->metadata.code_rows.ptr<const int *>(),
                           batch_->metadata.lane_rows.ptr<int>(),
                           batch_->metadata.caches.ptr<CacheSequence>()};
        model.depth_frame(*batch_->state, int(rows.size()), mapped, std::getenv("TTS_DUMP_PREFIX"));
        batch_->read_done(gpu, rows);
        for (const auto &row : rows) {
            auto &s = *row.request;
            s.metrics.prefill_ms = milliseconds(s.start);
            s.is_prefilled = true;
        }
    } catch (const DeviceError &) {
        throw;
    } catch (...) {
        for (const auto &row : rows) {
            row.request->failure = std::current_exception();
            row.request->finished = true;
        }
        prefill_.reset();
    }
    return rows.size();
}
void Runner::step(const std::vector<RequestState *> &requests,
                       const std::vector<AudioCallback> &callbacks) {
    if (requests.empty() || callbacks.size() != requests.size())
        throw std::invalid_argument("invalid execution batch");
    CUDA(cudaSetDevice(gpu.device));
    ExecutionScope execution(gpu);
    Range frame_range("tts.batch.frame");
    auto &batch = *batch_;
    std::vector<Batch::Row> decode, audio;
    bool graphs = true;
    for (auto *request : requests) {
        auto &s = state(*request);
        if (!s.is_prefilled)
            throw std::logic_error("decode scheduled before prefill completion");
        if (s.finished)
            continue;
        graphs &= s.opt.use_cuda_graph;
        if (s.frame) {
            auto lengths = s.prepared.lengths;
            for (auto &length : lengths)
                length += s.frame;
            try {
                s.kv_lease->ensure(lengths);
            } catch (const DeviceError &) {
                throw;
            } catch (...) {
                s.failure = std::current_exception();
                s.finished = true;
                continue;
            }
            decode.push_back({&s, 0, s.id});
        }
    }
    if (!decode.empty()) {
        int actual_lanes = 0;
        for (const auto &row : decode)
            actual_lanes += row.request->prepared.branches;
        // Shape planning is independent of launch mode, so eager/replay use
        // identical arithmetic and can be compared without changing GEMM geometry.
        auto shape = graph_shapes_.select(int(decode.size()), actual_lanes, max_guidance_branches);
        batch.map_decode(*this, decode, shape.first, shape.second);
        DecodeBatch view{batch.lanes, batch.metadata.sampling.ptr<SamplingRow>(),
                         batch.metadata.code_rows.ptr<const int *>(),
                         batch.metadata.lane_rows.ptr<int>(),
                         batch.metadata.caches.ptr<CacheSequence>()};
        auto execute = [&] { model.next_frame(*batch.state, batch.decode_capacity, view); };
        Range range("tts.batch.decode");
        batch.storage.graphs().run("decode", {shape.first, shape.second}, graphs, execute);
        // Sampling stores dense EOS flags alongside request-owned flags; one readback per batch.
        batch.read_done(gpu, decode);
        if (std::getenv("TTS_DUMP_PREFIX")) {
            int lane = 0;
            for (const auto &row : decode) {
                auto &s = *row.request;
                if (s.frame == 1 && row.lane == 0) {
                    TensorDebugScope debug_scope(s.id);
                    dump_tensor(gpu,
                                batch.state->last.view(batch.lanes, model.backbone.config.hidden)
                                    .slice(lane),
                                ".backbone_step1.bf16");
                }
                lane += s.prepared.branches;
            }
        }
    }
    for (auto *request : requests) {
        auto &s = state(*request);
        if (s.finished)
            continue;
        for (int lane = 0; lane < s.prepared.batch; ++lane)
            if (!s.ended[lane])
                audio.push_back({&s, lane, s.id});
        if (s.token_out.is_open()) {
            std::vector<int> codes(s.prepared.batch * model.config.codebooks);
            gpu.download(codes.data(), s.state->codes.data(), codes.size() * sizeof(int));
            s.token_out.write(reinterpret_cast<const char *>(codes.data()),
                              codes.size() * sizeof(int));
            if (!s.token_out)
                throw std::runtime_error("cannot write token debug output");
        }
    }
    if (!audio.empty()) {
        int shape = graph_shapes_.select(int(audio.size()));
        batch.map_codec(*this, audio, shape ? shape : int(audio.size()));
        CodecBatch view{batch.metadata.codec_codes.ptr<const int *>(),
                        batch.metadata.codec_caches.ptr<CacheSequence>(),
                        batch.metadata.histories.ptr<ConvHistory>(), batch.capacity};
        auto execute = [&] {
            model.codec.forward(batch.state->cs, nullptr, batch.audio_capacity, &view);
        };
        {
            Range range("tts.batch.codec");
            batch.storage.graphs().run("codec", {shape}, graphs, execute);
        }
        gpu.download(batch.pcm.data(), batch.state->cs.pcm.data(),
                     audio.size() * model.config.samples_per_frame * sizeof(int16_t));
    }
    size_t row = 0;
    for (size_t i = 0; i < requests.size(); ++i) {
        auto &s = state(*requests[i]);
        if (s.finished)
            continue;
        bool emitted = false;
        for (int lane = 0; lane < s.prepared.batch; ++lane) {
            if (s.ended[lane])
                continue;
            const auto *pcm = batch.pcm.data() + row++ * model.config.samples_per_frame;
            if (!s.delivered)
                s.metrics.first_audio_ms = milliseconds(s.start);
            ++s.delivered;
            emitted = true;
            s.metrics.audio_seconds +=
                double(model.config.samples_per_frame) / model.config.sample_rate;
            if (!s.cancelled && !callbacks[i](lane, pcm, model.config.samples_per_frame))
                s.cancelled = true;
        }
        if (emitted)
            s.metrics.frames = ++s.frame;
        s.finished = !emitted || s.cancelled || s.frame == s.opt.max_frames;
        s.metrics.truncated = !s.cancelled && s.frame == s.opt.max_frames;
        s.metrics.total_ms = milliseconds(s.start);
    }
    CUDA(cudaGetLastError());
}
Runner::State &Runner::state(RequestState &base) {
    auto &s = dynamic_cast<State &>(base);
    if (&s.owner != this)
        throw std::invalid_argument("request state belongs to another runner");
    return s;
}
std::shared_ptr<const Reference> Runner::prepare_reference(ReferencePlan &base) {
    auto &plan = dynamic_cast<ReferenceInput &>(base);
    return std::make_shared<const Reference>(std::move(plan.transcript), encode_audio(plan.mono),
                                             model.config.codebooks,
                                             model.config.codec.codebook_size);
}
std::vector<int> Runner::encode_audio(const std::vector<float> &mono) {
    CUDA(cudaSetDevice(gpu.device));
    ExecutionScope execution(gpu);
    if (mono.size() > size_t(model.config.max_reference_frames) * model.config.samples_per_frame)
        throw std::invalid_argument("reference audio exceeds context");
    DeviceArena scratch(gpu.memory, model.encoder.workspace_bytes(mono.size()));
    auto lease = ArenaLease(scratch);
    return model.encoder.encode(scratch, mono);
}
std::vector<float> Runner::decode_codes(const std::vector<int> &codes) {
    CUDA(cudaSetDevice(gpu.device));
    ExecutionScope execution(gpu);
    const auto &c = model.config;
    size_t frames = codes.size() / c.codebooks;
    if (codes.empty() || codes.size() % c.codebooks ||
        frames > size_t(c.codec.transformer.max_positions))
        throw std::runtime_error("codes must contain complete frames within codec position limits");
    for (int code : codes)
        if (code < 0 || code >= c.codec.codebook_size)
            throw std::runtime_error("codec ID out of range");
    WorkspacePlan plan;
    CodecState planned(model.codec, 1, &plan);
    size_t code_bytes = size_t(c.codebooks) * sizeof(int);
    plan.allocate(code_bytes);
    DeviceArena scratch(gpu.memory, plan.size);
    auto lease = ArenaLease(scratch);
    CodecState state(model.codec, 1, &scratch);
    state.reset();
    Buffer frame(code_bytes, &scratch);
    std::vector<float> pcm(frames * c.samples_per_frame);
    for (size_t t = 0; t < frames; t++) {
        gpu.upload(frame.data(), codes.data() + t * c.codebooks, code_bytes);
        TensorView out = model.codec.forward(state, frame.ptr<int>());
        gpu.download(pcm.data() + t * c.samples_per_frame, out.data, out.size());
    }
    return pcm;
}
} // namespace tts::breeze
