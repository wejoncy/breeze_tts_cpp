#pragma once
#include "model.h"
namespace tts {
struct GenerationState {
    TransformerState bs, ds;
    CodecState cs;
    Buffer x, last, logits, depth_hidden, depth_logits, codes, done, positions, rng, seen;
    GenerationState(const Model &m, int context, int b, int br, int rows, BufferAllocator *storage,
                    KVCacheView kv)
        : bs(m.backbone.config, b * br, context, rows, storage, kv),
          ds(m.depth.config, b * br, m.config.codebooks, b * br * 2, storage),
          cs(m.codec, b, storage),
          x(size_t(rows) * m.backbone.config.hidden * bytes(m.backbone.config.type), storage),
          last(size_t(b * br) * m.backbone.config.hidden * bytes(m.backbone.config.type), storage),
          logits(size_t(b * br) * m.lm_head.rows * bytes(m.lm_head.type), storage),
          depth_hidden(size_t(b * br) * 2 * m.depth.config.hidden * bytes(m.depth.config.type),
                       storage),
          depth_logits(size_t(b * br) * m.depth_heads[0].rows * bytes(m.depth_heads[0].type),
                       storage),
          codes(size_t(b) * m.config.codebooks * sizeof(int), storage),
          done(size_t(b) * 4, storage), positions(size_t(b * br) * 4, storage),
          rng(size_t(b) * 8, storage),
          seen(size_t(b) * m.config.codec.codebook_size * sizeof(int), storage) {}
};
} // namespace tts
