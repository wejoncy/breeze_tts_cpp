#include "ops/audio.h"
#include "models/breeze/prefill.h"
#include "models/breeze/model.h"
#include "models/breeze/prepared_batch.h"
#include "support/tensor_debug.h"
#include <cmath>
namespace tts::breeze {
namespace {
// Sizing and materialization share one layout. These buffers are only borrowed
// during prefill; captured decoder addresses belong to GenerationState.
struct PrefillState {
    TransformerState text, backbone;
    Buffer projected;
    Buffer ids, text_metadata, backbone_metadata, hidden;
    std::vector<Buffer> references;
    PrefillState(const Model &m, const PreparedBatch &p, BufferAllocator &storage, KVCacheView kv)
        : text(m.text.config, p.text_sequences, p.max_text, p.text_rows, &storage),
          backbone(m.backbone.config, p.lanes, p.max_prompt, p.storage_rows, &storage, kv),
          projected(size_t(p.prompt_rows) * m.backbone.config.hidden *
                        bytes(m.backbone.config.type),
                    &storage),
          ids(size_t(p.text_rows) * sizeof(int), &storage),
          text_metadata(p.text_metadata.size() * sizeof(int), &storage),
          backbone_metadata(p.backbone_metadata.size() * sizeof(int), &storage),
          hidden(size_t(p.text_rows) * m.text.config.hidden * bytes(m.text.config.type), &storage) {
        references.reserve(p.lanes);
        for (const auto &prompt : p.prompts)
            references.emplace_back(prompt.reference
                                        ? (prompt.reference->codes.size() + m.config.codebooks) *
                                              sizeof(int)
                                        : 0,
                                    &storage);
    }
};
} // namespace
size_t prefill_workspace_bytes(const Model &m, const PreparedBatch &p, KVCacheView kv) {
    WorkspacePlan plan;
    PrefillState state(m, p, plan, kv);
    return plan.size;
}

static PreparedBatch prefill_bound(const Model &m, int context, int branches) {
    PreparedBatch bound;
    bound.lanes = branches;
    bound.text_sequences = 2 * bound.lanes;
    bound.max_text = bound.max_prompt = context;
    bound.text_rows = bound.prompt_rows = bound.lanes * context;
    bound.storage_rows = (bound.prompt_rows + 127) / 128 * 128;
    bound.has_reference = true;
    bound.text_metadata.resize(2 * bound.text_rows + 2 * bound.text_sequences + 1);
    bound.backbone_metadata.resize(2 * bound.prompt_rows + 2 * bound.lanes + 1);
    auto reference = std::make_shared<Reference>();
    reference->codes.resize(size_t(context) * m.config.codebooks);
    bound.prompts.resize(bound.lanes);
    for (auto &prompt : bound.prompts)
        prompt.reference = reference;
    return bound;
}
size_t prefill_workspace_bound(const Model &m, int context, int branches, KVCacheView kv) {
    return prefill_workspace_bytes(m, prefill_bound(m, context, branches), kv);
}
size_t prefill_transfer_bound(const Model &m, int context, int branches) {
    WorkspacePlan sizing, transfers;
    PrefillState state(m, prefill_bound(m, context, branches), sizing, {});
    for (auto *b : {&state.ids, &state.text_metadata, &state.backbone_metadata})
        transfers.allocate(b->size());
    for (const auto &b : state.references) transfers.allocate(b.size());
    return transfers.size;
}

struct Prefill::Impl {
    const Model &model;
    const PreparedBatch &prepared;
    PrefillState buffers;
    TensorView last;
    SequenceLayout text_batch, backbone_batch;
    std::vector<std::vector<int>> reference_codes;
    enum class Stage { Embed, Text, Project, Backbone, Done } stage = Stage::Embed;
    int layer = 0;
    Impl(const Model &m, BufferAllocator &storage, const PreparedBatch &p,
         KVCacheView kv, TensorView output)
        : model(m), prepared(p), buffers(m, p, storage, kv), last(output),
          reference_codes(p.lanes) {
        auto &gpu = backend_for(storage.device_ordinal());
        for (int lane = 0; lane < p.lanes; ++lane)
            if (const auto &reference = p.prompts[lane].reference) {
                auto &codes = reference_codes[lane];
                codes = reference->codes;
                codes.resize(codes.size() + m.config.codebooks, m.config.codebook_eos);
            }
        gpu.upload(buffers.ids.data(), p.tokens.data(), buffers.ids.size());
        auto &tm = buffers.text_metadata;
        gpu.upload(tm.data(), p.text_metadata.data(), tm.size());
        text_batch = {tm.ptr<int>(), tm.ptr<int>() + p.text_rows, tm.ptr<int>() + 2 * p.text_rows,
                      tm.ptr<int>() + 2 * p.text_rows + p.text_sequences + 1, p.text_rows};
        text_batch.sequences = p.text_sequences;
        text_batch.max_length = p.max_text;
        backbone_batch = text_batch;
        if (p.has_reference) {
            auto &bm = buffers.backbone_metadata;
            gpu.upload(bm.data(), p.backbone_metadata.data(), bm.size());
            backbone_batch = {bm.ptr<int>(), bm.ptr<int>() + p.prompt_rows,
                              bm.ptr<int>() + 2 * p.prompt_rows,
                              bm.ptr<int>() + 2 * p.prompt_rows + p.lanes + 1, p.prompt_rows};
        }
        backbone_batch.sequences = p.lanes;
        backbone_batch.max_length = p.max_prompt;
    }
    bool step() {
        auto &gpu = backend_for(buffers.hidden.device());
        const auto &m = model;
        const auto &p = prepared;
        auto h = buffers.hidden.view(p.text_rows, m.text.config.hidden);
        auto projected = buffers.projected.view(p.prompt_rows, m.backbone.config.hidden);
        switch (stage) {
        case Stage::Embed:
            gpu.embedding(m.text_embedding, buffers.ids.ptr<int>(), h,
                          std::sqrt(float(m.text.config.hidden)), m.eoi, m.config.text_eoi);
            stage = Stage::Text;
            break;
        case Stage::Text: {
            Range range("tts.text_encoder.step");
            ForwardBatch input{p.text_sequences, p.max_text};
            input.fixed_position = 0;
            input.sequence = text_batch;
            input.cache = buffers.text.external_kv;
            m.text.forward(buffers.text, h, input, {layer, layer + 1});
            if (++layer == m.text.config.layers) {
                stage = Stage::Project;
                layer = 0;
            }
            break;
        }
        case Stage::Project:
            if (p.has_reference) {
                for (auto segment : p.segments)
                    gpu.linear(h.slice(segment.start, segment.size), m.text_proj,
                               projected.slice(segment.destination, segment.size));
            } else
                gpu.linear(h, m.text_proj, projected);
            for (int lane = 0; lane < p.lanes; ++lane)
                if (auto ref = p.prompts[lane].reference) {
                    const auto &codes = reference_codes[lane];
                    auto &input = buffers.references[lane];
                    gpu.upload(input.data(), codes.data(), input.size());
                    tts::audio_embedding(
                        gpu.stream, m.audio_embedding, input.ptr<int>(),
                        projected.slice(p.offsets[lane] + p.prompts[lane].reference_text.size(),
                                        codes.size() / m.config.codebooks),
                        codes.size() / m.config.codebooks, 1, m.config.codebooks);
                }
            dump_tensor(gpu, p.has_reference ? h : h.slice(0, p.lengths[0]),
                        p.has_reference ? ".text_segments.bf16" : ".text.bf16");
            dump_tensor(gpu, p.has_reference ? projected : projected.slice(0, p.lengths[0]),
                        p.has_reference ? ".reference_prompt.bf16" : ".projected.bf16");
            stage = Stage::Backbone;
            break;
        case Stage::Backbone: {
            Range range("tts.prefill.step");
            ForwardBatch input{p.lanes, p.max_prompt};
            input.fixed_position = 0;
            input.sequence = backbone_batch;
            input.cache = buffers.backbone.external_kv;
            input.output = last;
            m.backbone.forward(buffers.backbone, projected, input, {layer, layer + 1});
            if (++layer == m.backbone.config.layers) {
                dump_tensor(gpu, p.has_reference ? last : last.slice(0),
                            p.has_reference ? ".reference_backbone.bf16" : ".backbone.bf16");
                stage = Stage::Done;
            }
            break;
        }
        case Stage::Done:
            return true;
        }
        // Completion is a real scheduling checkpoint, not a growing GPU backlog.
        gpu.sync();
        return stage == Stage::Done;
    }
};
Prefill::Prefill(const Model &m, BufferAllocator &storage, const PreparedBatch &p,
                 KVCacheView kv, TensorView last)
    : impl_(std::make_unique<Impl>(m, storage, p, kv, last)) {}
Prefill::~Prefill() = default;
bool Prefill::step() {
    return impl_->step();
}
} // namespace tts::breeze
