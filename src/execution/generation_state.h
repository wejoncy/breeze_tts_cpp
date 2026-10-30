#pragma once
#include "models/breeze/model.h"
namespace tts {
struct GenerationState {
    TransformerState bs, ds;
    CodecState cs;
    Buffer last, logits, depth_hidden, depth_logits, done;
    GenerationState(const Model &m, int context, int b, int lanes, int rows,
                    BufferAllocator *storage, KVCacheView kv)
        : bs(m.backbone.config, lanes, context, rows, storage, kv),
          ds(m.depth.config, lanes, m.config.codebooks, lanes * 2, storage),
          cs(m.codec, b, storage, true),
          last(size_t(lanes) * m.backbone.config.hidden * bytes(m.backbone.config.type), storage),
          logits(size_t(lanes) * m.lm_head.rows * bytes(m.lm_head.type), storage),
          depth_hidden(size_t(lanes) * 2 * m.depth.config.hidden * bytes(m.depth.config.type),
                       storage),
          depth_logits(size_t(lanes) * m.depth_heads[0].rows * bytes(m.depth_heads[0].type),
                       storage),
          done(size_t(b) * sizeof(int), storage) {}
};
} // namespace tts
