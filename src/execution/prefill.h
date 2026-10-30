#pragma once
#include "core/kv_cache_view.h"
#include <memory>
namespace tts {
struct Model;
class Backend;
struct BufferAllocator;
struct PreparedBatch;
size_t prefill_workspace_bytes(const Model &, const PreparedBatch &, KVCacheView);
size_t prefill_workspace_bound(const Model &, int context, int branches, KVCacheView);
// Text attention is bidirectional, so a scheduling checkpoint is a whole
// transformer layer. Persist its activations while decode uses separate storage.
class Prefill {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    Prefill(const Model &, Backend &, BufferAllocator &, const PreparedBatch &, KVCacheView,
            TensorView last);
    ~Prefill();
    bool step();
};
} // namespace tts
