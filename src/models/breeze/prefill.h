#pragma once
#include "core/kv_cache_view.h"
#include <memory>
namespace tts {
class Backend;
struct BufferAllocator;
}
namespace tts::breeze {
struct Model;
struct PreparedBatch;
size_t prefill_workspace_bytes(const Model &, const PreparedBatch &, KVCacheView);
size_t prefill_workspace_bound(const Model &, int context, int branches, KVCacheView);
size_t prefill_transfer_bound(const Model &, int context, int branches);
// Text attention is bidirectional, so a scheduling checkpoint is a whole
// transformer layer. Persist its activations while decode uses separate storage.
class Prefill {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    Prefill(const Model &, BufferAllocator &, const PreparedBatch &, KVCacheView,
            TensorView last);
    ~Prefill();
    bool step();
};
} // namespace tts::breeze
