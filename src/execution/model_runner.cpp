#include "kernels/sampling/sampling.h"
#include "kernels/audio/audio_kernels.h"
#include "execution/model_runner.h"
#include "execution/generation_state.h"
#include "runtime/execution_resources.h"
#include "runtime/kv_cache.h"
#include "execution/prepared_batch.h"
#include "execution/request_planner.h"
#include "execution/prefill.h"
#include "support/tensor_debug.h"
#include "support/log.h"
#include <chrono>
#include <cmath>
#include <fstream>
#include <filesystem>
namespace tts {
using Clock = std::chrono::steady_clock;
static double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
struct ModelRunner::DecodeBatch {
    int lanes;
    const SamplingRow *sampling;
    const int *const *codes;
    const int *row_request;
    const CacheSequence *cache;
};
void ModelRunner::depth_frame(GenerationState &s, int batch, const DecodeBatch &mapped,
                              bool debug) {
    int lanes = mapped.lanes;
    auto &g = gpu;
    auto h = s.last.view(lanes, model.backbone.config.hidden),
         l = s.logits.view(lanes, model.lm_head.rows);
    {
        Range r("tts.lm_head");
        g.linear(h, model.lm_head, l);
        tts::sample(
            g.stream, l,
            {model.config.codebooks, model.config.codec.codebook_size, model.config.backbone_eos},
            mapped.sampling, batch, 0, s.done.ptr<int>());
    }
    if (debug)
        dump_tensor(g, l, ".backbone_logits.bf16");
    Range range("tts.depth");
    auto hidden = s.depth_hidden.view(lanes * 2, model.depth.config.hidden),
         dl = s.depth_logits.view(lanes, model.depth_heads[0].rows);
    auto first = hidden.every_rows(0, 2, lanes);
    auto dh = hidden.every_rows(1, 2, lanes);
    g.linear(h, model.depth_proj, first);
    tts::audio_embedding(g.stream, model.depth_embedding, nullptr, dh, lanes, 1,
                         model.config.codebooks, 0, mapped.codes, mapped.row_request);
    model.depth.forward(g, s.ds, hidden, nullptr, lanes, 2, 0, 0);
    for (int cb = 1; cb < model.config.codebooks; cb++) {
        if (cb > 1) {
            hidden = s.depth_hidden.view(lanes, model.depth.config.hidden);
            dh = hidden;
            tts::audio_embedding(g.stream, model.depth_embedding, nullptr, hidden, lanes, 1,
                                 model.config.codebooks, cb - 1, mapped.codes, mapped.row_request);
            model.depth.forward(g, s.ds, hidden, nullptr, lanes, 1, 0, cb);
        }
        g.linear(dh, model.depth_heads[cb - 1], dl);
        if (debug)
            dump_tensor(g, dl, ".depth_" + std::to_string(cb) + ".bf16");
        tts::sample(
            g.stream, dl,
            {model.config.codebooks, model.config.codec.codebook_size, model.config.backbone_eos},
            mapped.sampling, batch, cb, s.done.ptr<int>());
    }
}
void ModelRunner::next_frame(GenerationState &s, int batch, const DecodeBatch &mapped) {
    int lanes = mapped.lanes;
    {
        Range r("tts.backbone");
        auto input = s.last.view(lanes, model.backbone.config.hidden);
        tts::audio_embedding(gpu.stream, model.audio_embedding, nullptr, input, lanes, 1,
                             model.config.codebooks, -1, mapped.codes, mapped.row_request);
        model.backbone.forward(gpu, s.bs, input, nullptr, lanes, 1, 0, -1, {}, nullptr,
                               mapped.cache);
        gpu.increment(nullptr, lanes, 1, mapped.cache);
    }
    depth_frame(s, batch, mapped);
}
namespace {
// This model supports unconditional, instruction and reference guidance branches.
struct RequestBuffers {
    CodecHistory cs;
    Buffer codes, done, positions, rng, seen, table;
    RequestBuffers(const Model &m, int blocks, BufferAllocator *storage)
        : cs(m.codec, storage), codes(m.config.codebooks * sizeof(int), storage),
          done(sizeof(int), storage), positions(max_guidance_branches * sizeof(int), storage),
          rng(sizeof(uint64_t), storage), seen(m.config.codec.codebook_size * sizeof(int), storage),
          table(size_t(max_guidance_branches) * blocks * sizeof(int), storage) {}
};
} // namespace
struct ModelRunner::Slot {
    DeviceArena storage;
    std::unique_ptr<RequestBuffers> buffers;
    bool used = false;
    Slot(ModelRunner &r) : storage(r.gpu.memory) {
        WorkspacePlan plan;
        RequestBuffers layout(r.model, r.kv_->block_stride(), &plan);
        storage.reserve(plan.size);
        buffers = std::make_unique<RequestBuffers>(r.model, r.kv_->block_stride(), &storage);
    }
};
struct ModelRunner::PrefillWork {
    std::vector<uint64_t> ids;
    PreparedBatch prepared;
    std::vector<int> blocks;
    DeviceArena scratch;
    ArenaLease lease;
    Buffer table;
    Prefill state;
    static size_t workspace_bytes(ModelRunner &r, const PreparedBatch &p) {
        return prefill_workspace_bytes(r.model, p, r.kv_->view()) +
               BufferAllocator::aligned(size_t(p.lanes) * r.kv_->block_stride() * sizeof(int));
    }
    PrefillWork(ModelRunner &r, std::vector<uint64_t> identities, PreparedBatch p,
                std::vector<int> blocks);
};
struct ModelRunner::RequestState::Impl {
    ModelRunner &owner;
    Slot &slot;
    RequestBuffers *state;
    const GenerationOptions opt;
    const uint64_t id;
    PreparedBatch prepared;
    std::unique_ptr<PagedKVCache::Lease> kv_lease;
    std::vector<int> ended;
    std::ofstream token_out;
    Clock::time_point start = Clock::now();
    Stats stats;
    std::exception_ptr error;
    int frame = 0, delivered = 0;
    bool finished = false, cancelled = false, prefilled = false;
    Impl(ModelRunner &r, Slot &s, GenerationOptions options, uint64_t identity)
        : owner(r), slot(s), state(s.buffers.get()), opt(options), id(identity) {
        slot.used = true;
    }
    ~Impl() {
        if (owner.prefill_ && std::find(owner.prefill_->ids.begin(), owner.prefill_->ids.end(),
                                        id) != owner.prefill_->ids.end())
            owner.prefill_.reset();
        slot.used = false;
    }
};
// A single dense compute workspace, with borrowed pointers to request-owned state.
// Membership updates upload only small descriptors; KV, RNG and codec histories never move.
struct ModelRunner::Batch {
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
        RequestState::Impl *request;
        int lane;
        uint64_t id;
        bool operator==(const Row &other) const {
            return id == other.id && lane == other.lane;
        }
    };
    using Storage = ExecutionResources<std::pair<int, int>, Graphs>;
    Storage storage;
    std::unique_ptr<ArenaLease> lease;
    std::unique_ptr<GenerationState> state;
    Metadata metadata;
    std::vector<Row> decode_rows, audio_rows;
    std::vector<int16_t> pcm;
    std::vector<int> ended;
    int lanes = 0, decode_capacity = 0, audio_capacity = 0, capacity, lane_capacity;

    static size_t workspace_bytes(ModelRunner &r, int rows, int lanes) {
        WorkspacePlan plan;
        GenerationState planned(r.model, r.context, rows, lanes, lanes, &plan, r.kv_->view());
        Metadata metadata(rows, lanes, planned.cs.conv.size(), &plan);
        return plan.size;
    }
    Batch(ModelRunner &r, int rows, int lane_count)
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
    void map_decode(ModelRunner &r, const std::vector<Row> &rows, int padded_rows = 0,
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
        r.gpu.sync();
        lanes = int(sequences.size());
        decode_rows = rows;
    }
    void map_codec(ModelRunner &r, const std::vector<Row> &rows, int padded_rows) {
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
            for (size_t op = 0; op < cs.conv.size(); ++op)
                history[op * capacity + i] = {cs.conv[op].ptr<float>(),
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
        r.gpu.sync();
        audio_rows = rows;
    }
    void read_done(Backend &gpu, const std::vector<Row> &rows) {
        gpu.download(ended.data(), state->done.data(), rows.size() * sizeof(int));
        for (size_t i = 0; i < rows.size(); ++i)
            rows[i].request->ended[rows[i].lane] = ended[i];
    }
};
ModelRunner::PrefillWork::PrefillWork(ModelRunner &r, std::vector<uint64_t> identities,
                                      PreparedBatch p, std::vector<int> mapping)
    : ids(std::move(identities)), prepared(std::move(p)), blocks(std::move(mapping)),
      scratch(r.gpu.memory, workspace_bytes(r, prepared)), lease(scratch),
      table(blocks.size() * sizeof(int), &scratch),
      state(r.model, r.gpu, scratch, prepared, r.kv_->view(table.ptr<int>()),
            r.batch_->state->last.view(prepared.lanes, r.model.backbone.config.hidden)) {
    r.gpu.upload(table.data(), blocks.data(), table.size());
}
ModelRunner::ModelRunner(const Model &m, Backend &g, int batch, int ctx, int page,
                         size_t runtime_memory_bytes)
    : model(m), gpu(g), max_batch(batch), context(ctx), graph_shapes_(batch),
      kv_(std::make_unique<PagedKVCache>(
          g.memory, m.backbone.config.layers, m.backbone.config.kvheads * m.backbone.config.dim,
          m.backbone.config.type, batch * max_guidance_branches, ctx, page)) {
    slots_.reserve(max_batch);
    gpu.reserve_attention_workspace(size_t(batch) * max_guidance_branches * ctx *
                                    std::max(m.backbone.config.heads, 2 * m.text.config.heads));
    batch_ = std::make_unique<Batch>(*this, max_batch, max_batch * max_guidance_branches);
    for (int i = 0; i < max_batch; ++i)
        slots_.push_back(std::make_unique<Slot>(*this));
    // Capacity classes cover arbitrary prefill lengths without padding or an
    // exact-shape plan table. Configuration and library descriptors are init-only.
    gpu.begin_admission();
    auto prepare = [&](const char *name, const Transformer &transformer, int rows) {
        auto start = Clock::now();
        TTS_LOG(TTS_LOG_INFO, "engine.init", "planning start transformer={} max_rows={}", name,
                rows);
        transformer.prepare_range(gpu, rows);
        TTS_LOG(TTS_LOG_INFO, "engine.init", "planning complete transformer={} ms={:.0f}", name,
                milliseconds(start));
    };
    prepare("text", model.text, max_batch * max_guidance_branches * context);
    prepare("backbone", model.backbone, max_batch * max_guidance_branches * context);
    prepare("depth", model.depth, 2 * max_guidance_branches * max_batch);
    gpu.seal_plans();
    auto graph_start = Clock::now();
    TTS_LOG(TTS_LOG_INFO, "engine.init", "CUDA graph capture start");
    initialize_graphs();
    TTS_LOG(TTS_LOG_INFO, "engine.init", "CUDA graph capture complete ms={:.0f}",
            milliseconds(graph_start));

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
    gpu.memory.reserve_runtime(budget);
}
void ModelRunner::initialize_graphs() {
    auto &batch = *batch_;
    // Largest buckets first, sharing stable storage. Inactive rows are masked
    // inside existing kernels and own no KV pages, RNG or codec histories.
    const auto &buckets = graph_shapes_.rows();
    for (auto shape = buckets.rbegin(); shape != buckets.rend(); ++shape) {
        int rows = *shape;
        for (int branches = max_guidance_branches; branches >= 1; --branches) {
            int lanes = rows * branches;
            DecodeBatch view{lanes, batch.metadata.sampling.ptr<SamplingRow>(),
                             batch.metadata.code_rows.ptr<const int *>(),
                             batch.metadata.lane_rows.ptr<int>(),
                             batch.metadata.caches.ptr<CacheSequence>()};
            auto &graph = batch.storage.graphs_for({rows, lanes}).decoder;
            graph.capture(gpu.stream, [&] { next_frame(*batch.state, rows, view); });
        }
        CodecBatch view{batch.metadata.codec_codes.ptr<const int *>(),
                        batch.metadata.codec_caches.ptr<CacheSequence>(),
                        batch.metadata.histories.ptr<ConvHistory>(), batch.capacity};
        auto &graph = batch.storage.graphs_for({rows, 0}).codec;
        graph.capture(gpu.stream,
                      [&] { model.codec.forward(gpu, batch.state->cs, nullptr, rows, &view); });
    }
    batch.storage.seal();
}
ModelRunner::~ModelRunner() = default;
struct ModelRunner::RequestAdmission::Impl {
    ModelRunner &owner;
    std::multiset<size_t>::iterator prefill;
    std::vector<std::multiset<int>::iterator> kv;
    Impl(ModelRunner &r, size_t workspace, const std::vector<int> &pages, int peak_pages)
        : owner(r) {
        prefill = r.prefill_credits_.insert(workspace);
        try {
            kv.reserve(pages.size());
            for (int count : pages)
                kv.push_back(r.kv_credits_.insert(count));
            r.kv_->reserve(peak_pages);
        } catch (...) {
            for (auto credit : kv)
                r.kv_credits_.erase(credit);
            r.prefill_credits_.erase(prefill);
            throw;
        }
    }
    ~Impl() {
        owner.prefill_credits_.erase(prefill);
        for (auto credit : kv)
            owner.kv_credits_.erase(credit);
    }
};
ModelRunner::RequestAdmission::RequestAdmission(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
ModelRunner::RequestAdmission::~RequestAdmission() = default;
std::unique_ptr<ModelRunner::RequestAdmission> ModelRunner::admit(RequestPlan &plan) try {
    CUDA(cudaSetDevice(gpu.device));
    Range admission("tts.batch.admission");
    std::vector<int> pages;
    size_t workspace = 0;
    for (const auto &input : plan.inputs) {
        pages.push_back(kv_->required_pages(input.capacities));
        workspace = std::max(workspace, PrefillWork::workspace_bytes(*this, input));
    }
    // At most max_batch inputs can execute across all admitted requests. Protect
    // the largest such subset, rather than summing each request's private peak.
    // This also covers every later input wave without another resource admission.
    std::vector<int> simultaneous = pages;
    simultaneous.insert(simultaneous.end(), kv_credits_.begin(), kv_credits_.end());
    std::sort(simultaneous.begin(), simultaneous.end(), std::greater<int>());
    int peak_pages = 0;
    for (size_t i = 0; i < std::min(simultaneous.size(), size_t(max_batch)); ++i)
        peak_pages += simultaneous[i];
    size_t kv_bytes;
    try {
        kv_bytes = kv_->additional_bytes(std::max(0, peak_pages - kv_->committed()));
    } catch (const std::invalid_argument &) {
        throw RequestOverloaded("request KV credits exceed available pool capacity");
    }
    const size_t protected_workspace =
        prefill_credits_.empty() ? workspace : std::max(workspace, *prefill_credits_.rbegin());
    const size_t additional = kv_bytes + BufferAllocator::aligned(protected_workspace);
    const size_t available = gpu.memory.available_bytes();
    if (additional > available || !gpu.memory.can_allocate({kv_bytes, protected_workspace})) {
        constexpr size_t mib = 1 << 20;
        throw RequestOverloaded("insufficient GPU memory: request needs " +
                                std::to_string((additional + mib - 1) / mib) +
                                " MiB additional, available " + std::to_string(available / mib) +
                                " MiB");
    }
    return std::unique_ptr<RequestAdmission>(new RequestAdmission(
        std::make_unique<RequestAdmission::Impl>(*this, workspace, pages, peak_pages)));
} catch (const DeviceOutOfMemory &error) {
    throw RequestOverloaded(error.what());
}
std::unique_ptr<ModelRunner::RequestState> ModelRunner::prepare(RequestPlan &plan, size_t lane,
                                                                RequestAdmission &admission) try {
    CUDA(cudaSetDevice(gpu.device));
    Range preparation("tts.batch.admission");
    if (&admission.impl_->owner != this)
        throw std::invalid_argument("request admission belongs to another runner");
    auto prepared = std::move(plan.inputs.at(lane));
    auto options = plan.options;
    options.seed = request_seed(options.seed, lane);
    auto found = std::find_if(slots_.begin(), slots_.end(), [](const auto &s) { return !s->used; });
    if (found == slots_.end())
        throw RequestOverloaded("request state capacity exhausted");
    auto impl = std::make_unique<RequestState::Impl>(*this, **found, options, ++request_sequence_);
    auto &s = *impl;
    s.prepared = std::move(prepared);
    s.ended.resize(s.prepared.batch);
    TensorDebugScope debug_scope(s.id);
    s.kv_lease = std::make_unique<PagedKVCache::Lease>(
        *kv_, s.prepared.lengths, s.prepared.capacities, s.state->table.ptr<int>());
    gpu.zero(s.state->codes);
    gpu.zero(s.state->done);
    gpu.zero(s.state->seen);
    s.state->cs.reset(gpu);
    gpu.upload(s.state->rng.data(), &s.opt.seed, sizeof(uint64_t));
    gpu.upload(s.state->positions.data(), s.prepared.lengths.data(),
               s.prepared.lengths.size() * sizeof(int));
    if (const char *path = std::getenv("TTS_TOKENS_OUT")) {
        s.token_out.open(std::filesystem::u8path(debug_path(path, s.id)), std::ios::binary);
        if (!s.token_out)
            throw std::runtime_error("cannot create token debug output");
    }
    return std::unique_ptr<RequestState>(new RequestState(std::move(impl)));
} catch (const DeviceOutOfMemory &error) {
    // Admission failure does not alter another request's state or grow backing.
    throw RequestOverloaded(error.what());
}
size_t ModelRunner::prefill_step(const std::vector<RequestState *> &requests) {
    CUDA(cudaSetDevice(gpu.device));
    if (!prefill_) {
        std::vector<const PreparedBatch *> inputs;
        std::vector<uint64_t> ids;
        std::vector<int> blocks;
        PreparedBatch packed;
        for (auto *request : requests) {
            auto &s = *request->impl_;
            if (s.prefilled || s.finished)
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
                                  [&](const auto *request) { return request->impl_->id == id; });
        if (found == requests.end())
            throw std::logic_error("active prefill input missing from execution batch");
        rows.push_back({(*found)->impl_.get(), 0, id});
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
        depth_frame(*batch_->state, int(rows.size()), mapped, std::getenv("TTS_DUMP_PREFIX"));
        batch_->read_done(gpu, rows);
        for (const auto &row : rows) {
            auto &s = *row.request;
            s.stats.prefill_ms = milliseconds(s.start);
            s.prefilled = true;
        }
    } catch (const DeviceError &) {
        throw;
    } catch (...) {
        for (const auto &row : rows) {
            row.request->error = std::current_exception();
            row.request->finished = true;
        }
        prefill_.reset();
    }
    return rows.size();
}
void ModelRunner::step(const std::vector<RequestState *> &requests,
                       const std::vector<AudioCallback> &callbacks) {
    if (requests.empty() || callbacks.size() != requests.size())
        throw std::invalid_argument("invalid execution batch");
    CUDA(cudaSetDevice(gpu.device));
    Range frame_range("tts.batch.frame");
    auto &batch = *batch_;
    std::vector<Batch::Row> decode, audio;
    bool graphs = true;
    for (auto *request : requests) {
        auto &s = *request->impl_;
        if (!s.prefilled)
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
                s.error = std::current_exception();
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
        auto shape = graph_shapes_.decoder(int(decode.size()), actual_lanes, max_guidance_branches);
        batch.map_decode(*this, decode, shape.first, shape.second);
        DecodeBatch view{batch.lanes, batch.metadata.sampling.ptr<SamplingRow>(),
                         batch.metadata.code_rows.ptr<const int *>(),
                         batch.metadata.lane_rows.ptr<int>(),
                         batch.metadata.caches.ptr<CacheSequence>()};
        auto execute = [&] { next_frame(*batch.state, batch.decode_capacity, view); };
        auto *prepared_graph = batch.storage.find_graphs(shape);
        Graph *graph = graphs && prepared_graph ? &prepared_graph->decoder : nullptr;
        Range range("tts.batch.decode");
        if (graph)
            graph->run(gpu.stream);
        else
            execute();
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
        auto &s = *request->impl_;
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
        int shape = graph_shapes_.codec(int(audio.size()));
        batch.map_codec(*this, audio, shape ? shape : int(audio.size()));
        CodecBatch view{batch.metadata.codec_codes.ptr<const int *>(),
                        batch.metadata.codec_caches.ptr<CacheSequence>(),
                        batch.metadata.histories.ptr<ConvHistory>(), batch.capacity};
        auto execute = [&] {
            model.codec.forward(gpu, batch.state->cs, nullptr, batch.audio_capacity, &view);
        };
        auto *prepared_graph = batch.storage.find_graphs({shape, 0});
        Graph *graph = graphs && prepared_graph ? &prepared_graph->codec : nullptr;
        {
            Range range("tts.batch.codec");
            if (graph)
                graph->run(gpu.stream);
            else
                execute();
        }
        gpu.download(batch.pcm.data(), batch.state->cs.pcm.data(),
                     audio.size() * model.config.samples_per_frame * sizeof(int16_t));
    }
    size_t row = 0;
    for (size_t i = 0; i < requests.size(); ++i) {
        auto &s = *requests[i]->impl_;
        if (s.finished)
            continue;
        bool emitted = false;
        for (int lane = 0; lane < s.prepared.batch; ++lane) {
            if (s.ended[lane])
                continue;
            const auto *pcm = batch.pcm.data() + row++ * model.config.samples_per_frame;
            if (!s.delivered)
                s.stats.first_audio_ms = milliseconds(s.start);
            ++s.delivered;
            emitted = true;
            s.stats.audio_seconds +=
                double(model.config.samples_per_frame) / model.config.sample_rate;
            if (!s.cancelled && !callbacks[i](lane, pcm, model.config.samples_per_frame))
                s.cancelled = true;
        }
        if (emitted)
            s.stats.frames = ++s.frame;
        s.finished = !emitted || s.cancelled || s.frame == s.opt.max_frames;
        s.stats.truncated = !s.cancelled && s.frame == s.opt.max_frames;
        s.stats.total_ms = milliseconds(s.start);
    }
    CUDA(cudaGetLastError());
}
ModelRunner::RequestState::RequestState(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
ModelRunner::RequestState::~RequestState() = default;
bool ModelRunner::RequestState::done() const {
    return impl_->finished;
}
bool ModelRunner::RequestState::prefilled() const {
    return impl_->prefilled;
}
Stats ModelRunner::RequestState::stats() const {
    return impl_->stats;
}
std::exception_ptr ModelRunner::RequestState::error() const {
    return impl_->error;
}
std::vector<int> ModelRunner::encode_audio(const std::vector<float> &mono) {
    CUDA(cudaSetDevice(gpu.device));
    if (mono.size() > size_t(model.config.max_reference_frames) * model.config.samples_per_frame)
        throw std::invalid_argument("reference audio exceeds context");
    DeviceArena scratch(gpu.memory, model.encoder.workspace_bytes(mono.size()));
    auto lease = ArenaLease(scratch);
    return model.encoder.encode(gpu, scratch, mono);
}
std::vector<float> ModelRunner::decode_codes(const std::vector<int> &codes) {
    CUDA(cudaSetDevice(gpu.device));
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
    state.reset(gpu);
    Buffer frame(code_bytes, &scratch);
    std::vector<float> pcm(frames * c.samples_per_frame);
    for (size_t t = 0; t < frames; t++) {
        gpu.upload(frame.data(), codes.data() + t * c.codebooks, code_bytes);
        TensorView out = model.codec.forward(gpu, state, frame.ptr<int>());
        gpu.download(pcm.data() + t * c.samples_per_frame, out.data, out.size());
    }
    return pcm;
}
} // namespace tts
