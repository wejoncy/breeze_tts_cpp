#include "audio_kernels.h"
#include "prefill.h"
#include "generation_state.h"
#include "prepared_batch.h"
#include "tensor_debug.h"
#include <cmath>
namespace tts {
namespace {
// Sizing and materialization share one layout. These buffers are only borrowed
// during prefill; captured decoder addresses belong to GenerationState.
struct PrefillState {
    TransformerState text;
    Buffer ids, text_metadata, backbone_metadata, hidden, zero;
    std::vector<Buffer> references;
    PrefillState(const Model &m, const PreparedBatch &p, BufferAllocator &storage)
        : text(m.text.config, p.text_sequences, p.max_text, p.text_rows, &storage),
          ids(size_t(p.text_rows) * sizeof(int), &storage),
          text_metadata(p.lanes > 1 || p.has_reference ? p.text_metadata.size() * sizeof(int) : 0,
                        &storage),
          backbone_metadata(p.backbone_metadata.size() * sizeof(int), &storage),
          hidden(size_t(p.text_rows) * m.text.config.hidden * bytes(m.text.config.type), &storage),
          zero(p.lanes == 1 && !p.has_reference ? sizeof(int) : 0, &storage) {
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
size_t prefill_workspace_bytes(const Model &m, const PreparedBatch &p) {
    WorkspacePlan plan;
    PrefillState state(m, p, plan);
    return plan.size;
}

void prefill(const Model &m, Backend &gpu, BufferAllocator &storage, const PreparedBatch &p,
             GenerationState &s) {
    auto &text = m.text;
    auto &backbone = m.backbone;
    const int lanes = p.lanes;
    PrefillState buffers(m, p, storage);
    auto &ts = buffers.text;
    auto &ids = buffers.ids, &hidden = buffers.hidden, &zero = buffers.zero;
    auto &tm = buffers.text_metadata, &bm = buffers.backbone_metadata;
    if (!p.has_reference && lanes == 1) {
        int n = p.max_prompt;
        gpu.zero(zero);
        gpu.upload(ids.data(), p.tokens.data(), ids.size());
        auto h = hidden.view(n, text.config.hidden),
             projected = s.x.view(n, backbone.config.hidden);
        {
            Range range("tts.text_encoder");
            gpu.embedding(m.text_embedding, ids.ptr<int>(), h, std::sqrt(float(text.config.hidden)),
                          m.eoi, m.config.text_eoi);
            text.forward(gpu, ts, h, zero.ptr<int>(), 1, n, 0, 0);
            gpu.linear(h, m.text_proj, projected);
        }
        dump_tensor(gpu, h, ".text.bf16");
        dump_tensor(gpu, projected, ".projected.bf16");
        {
            Range range("tts.prefill");
            backbone.forward(gpu, s.bs, projected, zero.ptr<int>(), 1, n, 0, 0,
                             s.last.view(1, backbone.config.hidden));
        }
        dump_tensor(gpu, s.last.view(1, backbone.config.hidden), ".backbone.bf16");
        gpu.sync();
    } else {
        gpu.upload(ids.data(), p.tokens.data(), ids.size());
        gpu.upload(tm.data(), p.text_metadata.data(), tm.size());
        if (p.has_reference)
            gpu.upload(bm.data(), p.backbone_metadata.data(), bm.size());
        PackedBatch tp{tm.ptr<int>(), tm.ptr<int>() + p.text_rows, tm.ptr<int>() + 2 * p.text_rows,
                       tm.ptr<int>() + 2 * p.text_rows + p.text_sequences + 1, p.text_rows};
        PackedBatch bp = tp;
        if (p.has_reference)
            bp = {bm.ptr<int>(), bm.ptr<int>() + p.prompt_rows, bm.ptr<int>() + 2 * p.prompt_rows,
                  bm.ptr<int>() + 2 * p.prompt_rows + lanes + 1, p.prompt_rows};
        auto h = hidden.view(p.text_rows, text.config.hidden),
             projected = s.x.view(p.prompt_rows, backbone.config.hidden);
        {
            Range range("tts.text_encoder");
            gpu.embedding(m.text_embedding, ids.ptr<int>(), h, std::sqrt(float(text.config.hidden)),
                          m.eoi, m.config.text_eoi);
            text.forward(gpu, ts, h, nullptr, p.text_sequences, p.max_text, 0, 0, {}, &tp);
            if (p.has_reference) {
                for (auto segment : p.segments)
                    gpu.linear(h.slice(segment.start, segment.size), m.text_proj,
                               projected.slice(segment.destination, segment.size));
            } else
                gpu.linear(h, m.text_proj, projected);
        }
        for (int lane = 0; lane < lanes; ++lane)
            if (auto ref = p.prompts[lane].reference) {
                auto codes = ref->codes;
                codes.resize(codes.size() + m.config.codebooks, m.config.codebook_eos);
                auto &input = buffers.references[lane];
                gpu.upload(input.data(), codes.data(), input.size());
                tts::audio_embedding(
                    gpu.stream, m.audio_embedding, input.ptr<int>(),
                    projected.slice(p.offsets[lane] + p.prompts[lane].reference_text.size(),
                                    codes.size() / m.config.codebooks),
                    codes.size() / m.config.codebooks, 1, m.config.codebooks);
                gpu.sync();
            }
        dump_tensor(gpu, p.has_reference ? h : h.slice(0, p.lengths[0]),
                    p.has_reference ? ".text_segments.bf16" : ".text.bf16");
        dump_tensor(gpu, p.has_reference ? projected : projected.slice(0, p.lengths[0]),
                    p.has_reference ? ".reference_prompt.bf16" : ".projected.bf16");
        {
            Range range("tts.prefill");
            backbone.forward(gpu, s.bs, projected, nullptr, lanes, p.max_prompt, 0, 0,
                             s.last.view(lanes, backbone.config.hidden), &bp);
        }
        auto last = s.last.view(lanes, backbone.config.hidden);
        dump_tensor(gpu, p.has_reference ? last : last.slice(0),
                    p.has_reference ? ".reference_backbone.bf16" : ".backbone.bf16");
    }
    gpu.upload(s.positions.data(), p.lengths.data(), p.lengths.size() * 4);
    gpu.sync();
}
} // namespace tts
