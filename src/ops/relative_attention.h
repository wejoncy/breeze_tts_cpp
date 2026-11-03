#pragma once
#include "core/tensor.h"
#include "core/sequence_layout.h"
#include "ops/attention.h"

namespace tts {
class Backend;
namespace ops {
size_t relative_attention_workspace_bytes(int batch, int time, int heads, int head_dim, Type);
// Scratch is borrowed from the execution arena; QKV is consumed in place.
void relative_attention(TensorView qkv, TensorView bias, TensorView relative_keys,
                        TensorView relative_values, TensorView output, TensorView scratch,
                        int batch, int time, int heads, int window,
                        const SequenceLayout *packed = nullptr);
} // namespace ops
} // namespace tts
