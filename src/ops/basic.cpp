#include "ops/basic.h"
#include "ops/storage.h"
#include "ops/sequence.h"
#include "ops/sampling.h"
#include "ops/cuda/backend.h"

namespace tts::ops {
void prepare_matmul(TensorView input, TensorView weight, TensorView output, bool full_precision,
                      float accumulation, float product_scale, int groups, int max_rows) {
    execution_context(input).prepare_linear(input, weight, output, full_precision, accumulation,
                                       product_scale, groups, max_rows);
}
void reserve_attention(int device, size_t elements) { execution_context(device).reserve_attention_workspace(elements); }
int decode_capacity(int device, int head_dimensions, Type type) {
    return execution_context(device).flash_decode_capacity(head_dimensions, type);
}
void reserve_decode(int device, int head_dimensions, Type type) {
    execution_context(device).reserve_flash_decode_workspace(head_dimensions, type);
}
void seal(int device) { execution_context(device).seal_plans(); }
void zero(TensorView tensor) {
    if (!tensor.dense()) throw std::invalid_argument("zero requires dense storage");
    CUDA(cudaMemsetAsync(tensor.data, 0, tensor.size(), execution_context(tensor).stream));
}
bool prepare_matmul_gelu(TensorView input, TensorView weight, TensorView bias, TensorView output,
                          int max_rows) {
    return execution_context(input).prepare_linear_gelu(input, weight, bias, output, max_rows);
}
bool matmul_gelu(TensorView input, TensorView weight, TensorView bias, TensorView output) {
    return execution_context(input).linear_gelu(input, weight, bias, output);
}
bool matmul_qkv(TensorView input, TensorView weight, const QkvProjectionOutput &output) {
    return execution_context(input).linear_qkv(input, weight, output);
}
void prepare_quantized(TensorView weight, int rows, int output_channels, int input_channels,
                        Type type, const Epilogue &epilogue) {
    auto &quantized = execution_context(weight).quantized();
    quantized.prepare(rows, output_channels, input_channels, type);
    quantized.prepare_finish(rows, output_channels, input_channels, type, epilogue);
}
void norm(TensorView input, TensorView weight, TensorView output, float epsilon, bool gemma,
            TensorView bias) {
    execution_context(input).norm(input, weight, output, epsilon, gemma, bias);
}
void residual_norm(TensorView input, TensorView update, TensorView weight, TensorView output,
                     float epsilon, TensorView scale, bool gemma) {
    execution_context(input).residual_norm(input, update, weight, output, epsilon, scale, gemma);
}
void residual_norm_last(TensorView input, TensorView update, TensorView weight, TensorView output,
                          const int *ends, float epsilon) {
    execution_context(input).residual_norm_last(input, update, weight, output, ends, epsilon);
}
void quantized_norm(TensorView input, TensorView update, TensorView weight, RowScaledView output,
                      float epsilon, bool gemma, TensorView scale) {
    execution_context(input).quantized().norm(input, update, weight, output, epsilon, gemma, scale);
}
void quantized_gated(TensorView input, RowScaledView output, int kind) {
    execution_context(input).quantized().gated(input, output, kind);
}
void add(TensorView input, TensorView update, TensorView output, TensorView scale) {
    execution_context(input).add(input, update, output, scale);
}
void bias(TensorView input, TensorView bias) { execution_context(input).bias(input, bias); }
void activation(TensorView input, TensorView output, int kind) {
    execution_context(input).activation(input, output, kind);
}
void gated_activation(TensorView input, TensorView output, int kind) {
    execution_context(input).gated(input, output, kind);
}
void write_qkv(TensorView input, TensorView query, TensorView key, TensorView value,
                 const int *positions, int batch, int time, int slot, int capacity, int heads,
                 int key_heads, int dimensions, TensorView query_norm, TensorView key_norm,
                 float epsilon, bool gemma, TensorView rope, int fixed_position,
                 const SequenceLayout *sequence, CacheLayout cache, TensorView bias) {
    execution_context(input).qkv(input, query, key, value, positions, batch, time, slot, capacity,
                            heads, key_heads, dimensions, query_norm, key_norm, epsilon, gemma,
                            rope, fixed_position, sequence, cache, bias);
}
void attention(TensorView query, TensorView key, TensorView value, TensorView output,
                 const int *positions, int batch, int time, int slot, int capacity, int heads,
                 int key_heads, int dimensions, bool causal, int window, int fixed_position,
                 const SequenceLayout *sequence, CacheLayout cache) {
    execution_context(query).attention(query, key, value, output, positions, batch, time, slot, capacity,
                                  heads, key_heads, dimensions, causal, window, fixed_position,
                                  sequence, cache);
}
void advance_positions(TensorView anchor, int *positions, int count, int value,
                         const CacheSequence *sequences) {
    execution_context(anchor).increment(positions, count, value, sequences);
}
int normalized_linear_capacity(TensorView input, bool qkv_writeback) {
    return execution_context(input).normalized_linear_capacity(qkv_writeback);
}
void grouped_matmul(TensorView input, TensorView weight, TensorView output, int groups) {
    execution_context(input, {weight, output}).grouped_linear(input, weight, output, groups);
}
void matmul(TensorView input, TensorView weight, TensorView output, bool full_precision,
             float accumulation, float product_scale) {
    execution_context(input, {weight, output}).linear(input, weight, output, full_precision,
                                               accumulation, product_scale);
}
void matmul(RowScaledView input, RowScaledView weight, TensorView output) {
    execution_context(output).quantized().linear(input, weight, output);
}
void lookup(TensorView table, const int *ids, TensorView output, float scale,
             TensorView replacement, int replacement_id) {
    execution_context(table).embedding(table, ids, output, scale, replacement, replacement_id);
}
void rotary_table(TensorView table, int dimensions, float theta, float factor, int kind,
                   float low_wave, float high_wave) {
    execution_context(table).make_rope(table, dimensions, theta, factor, kind, low_wave, high_wave);
}
void upload_bytes(int device, void *destination, const void *source, size_t bytes) {
    execution_context(device).upload(destination, source, bytes);
}
void synchronize(int device) { execution_context(device).sync(); }
void download_bytes(TensorView source, void *destination, size_t bytes) {
    execution_context(source).download(destination, source.data, bytes);
}
void quantize_rows(TensorView input, RowScaledView output) {
    execution_context(input).quantized().quantize(input, output);
}
} // namespace tts::ops
