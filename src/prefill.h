#pragma once
#include <cstddef>
namespace tts {
struct Model;
class Backend;
struct BufferAllocator;
struct PreparedBatch;
struct GenerationState;
size_t prefill_workspace_bytes(const Model &, const PreparedBatch &);
void prefill(const Model &, Backend &, BufferAllocator &, const PreparedBatch &, GenerationState &);
} // namespace tts
