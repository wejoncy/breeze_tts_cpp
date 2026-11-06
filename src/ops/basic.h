#pragma once
#include "core/tensor.h"
#include "core/quantization.h"
#include "core/qkv_projection.h"
#include "core/forward_batch.h"
#include "ops/sequence.h"

namespace tts::ops {
void matmul(TensorView input, TensorView weight, TensorView output, bool full_precision = false,
             float accumulation = 0, float product_scale = 1);
void matmul(RowScaledView input, RowScaledView weight, TensorView output);
void grouped_matmul(TensorView input, TensorView weight, TensorView output, int groups);
void prepare_matmul(TensorView input, TensorView weight, TensorView output,
                      bool full_precision = false, float accumulation = 0, float product_scale = 1,
                      int groups = 1, int max_rows = 0);
void reserve_attention(int device, size_t elements);
int decode_capacity(int device, int head_dimensions, Type type);
void reserve_decode(int device, int head_dimensions, Type type);
void seal(int device);
void zero(TensorView tensor);
bool prepare_matmul_gelu(TensorView input, TensorView weight, TensorView bias, TensorView output,
                          int max_rows);
bool matmul_gelu(TensorView input, TensorView weight, TensorView bias, TensorView output);
bool matmul_qkv(TensorView input, TensorView weight, const QkvProjectionOutput &output);
void prepare_quantized(TensorView weight, int rows, int output_channels, int input_channels,
                        Type type, const Epilogue &epilogue = {}, QuantizationGroups groups = {});
void norm(TensorView input, TensorView weight, TensorView output, float epsilon, bool gemma = false,
            TensorView bias = {});
void residual_norm(TensorView input, TensorView update, TensorView weight, TensorView output,
                     float epsilon, TensorView scale = {}, bool gemma = false);
void residual_norm_last(TensorView input, TensorView update, TensorView weight, TensorView output,
                          const int *ends, float epsilon);
void quantized_norm(TensorView input, TensorView update, TensorView weight, RowScaledView output,
                      float epsilon, bool gemma = false, TensorView scale = {});
void quantized_gated(TensorView input, RowScaledView output, int kind);
void add(TensorView input, TensorView update, TensorView output, TensorView scale = {});
void bias(TensorView input, TensorView bias);
void activation(TensorView input, TensorView output, int kind);
void gated_activation(TensorView input, TensorView output, int kind);
void write_qkv(TensorView input, TensorView query, TensorView key, TensorView value,
                 const int *positions, int batch, int time, int slot, int capacity, int heads,
                 int key_heads, int dimensions, TensorView query_norm, TensorView key_norm,
                 float epsilon, bool gemma, TensorView rope, int fixed_position = -1,
                 const SequenceLayout *sequence = nullptr, CacheLayout cache = {}, TensorView bias = {});
void attention(TensorView query, TensorView key, TensorView value, TensorView output,
                 const int *positions, int batch, int time, int slot, int capacity, int heads,
                 int key_heads, int dimensions, bool causal, int window, int fixed_position = -1,
                 const SequenceLayout *sequence = nullptr, CacheLayout cache = {});
void advance_positions(TensorView anchor, int *positions, int count, int value = 1,
                         const CacheSequence *sequences = nullptr);
int normalized_linear_capacity(TensorView input, bool qkv_writeback = false);
void lookup(TensorView table, const int *ids, TensorView output, float scale,
             TensorView replacement = {}, int replacement_id = -1);
void rotary_table(TensorView table, int dimensions, float theta, float factor, int kind,
                   float low_wave = 0, float high_wave = 0);
} // namespace tts::ops
