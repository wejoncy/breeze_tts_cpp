#pragma once
#include "core/qkv_projection.h"
#include "core/quantization.h"
#include "core/sequence_layout.h"
#include <cuda_runtime_api.h>

namespace tts::ops {
enum class Activation { Identity, Relu, LeakyRelu, Silu, Gelu, GeluTanh, Mish, Tanh, Sigmoid };
struct Epilogue {
    TensorView bias, residual;
    float value_scale = 1;
    float output_scale = 1;
    float residual_scale = 1;
    Activation activation = Activation::Identity;
    float alpha = .1f;
    // Optional pre-activation result, produced by the same kernel.
    TensorView raw;
    TensorView gate;
    TensorView condition;
    int rows_per_sequence = 1;
    int valid_rows_per_sequence = 0; // zero disables sequence-tail masking
    int input_repeat = 1;
    // Index into residual rows; negative entries contribute zero.
    const int *residual_rows = nullptr;
    const SequenceLayout *packed = nullptr; // Host view, copied into kernel arguments.
    const void *const *condition_sources = nullptr; // One borrowed channel row per sequence.
    void *const *output_rows = nullptr; // Pointwise-only final row destinations.
};
struct ConvGeometry {
    int input_channels, output_channels, kernel, stride = 1, padding = 0, dilation = 1, groups = 1;
    bool transpose = false;
    int padding_right = -1; // -1 mirrors the left padding.
    int output_length(int input) const;
};
void pointwise(cudaStream_t, TensorView x, TensorView y, const Epilogue & = {});
// Quantize the storage-rounded result in the same pass. Retained is optional;
// raw keeps its usual pre-activation meaning, independently of retained.
// Pass the initialization-time SM count to spread short grids across more warps.
void pointwise_quantize(cudaStream_t, TensorView x, RowScaledView quantized,
                        const Epilogue & = {}, TensorView retained = {}, int multiprocessors = 0);
// Also used when a producer applies the epilogue before quantization.
TensorShape validate_pointwise_quantize(TensorView x, RowScaledView quantized,
                                       const Epilogue & = {}, TensorView retained = {});
// Linear interpolation along time, with half-pixel centers (align_corners=false).
void linear_resample(cudaStream_t, TensorView x, TensorView y, int batch, int input_time,
                     int output_time, const SequenceLayout *input_layout = nullptr,
                     const SequenceLayout *output_layout = nullptr);
void convolution_finish(cudaStream_t, TensorView projected, TensorView output, int batch,
                        int input_length, const ConvGeometry &, const Epilogue &,
                        const SequenceLayout *input_layout = nullptr, const SequenceLayout *output_layout = nullptr);
void direct_convolution(cudaStream_t, TensorView x, TensorView weight, TensorView output, int batch,
                        int input_length, const ConvGeometry &, const Epilogue &,
                        const SequenceLayout *input_layout = nullptr, const SequenceLayout *output_layout = nullptr);
// Residual and projection bias are added before normalization. The optional raw
// result preserves that sum for the next residual branch without another pass.
struct NormFusion {
    TensorView residual, projection_bias, raw, gate, shift, scale;
    int rows_per_sequence = 1;
    const int *source_rows = nullptr;
};
void layer_norm(cudaStream_t, TensorView x, TensorView y, TensorView weight, TensorView bias,
                float eps, const NormFusion & = {});
// Retain the normalized 16-bit residual only when its later consumer needs it.
void layer_norm_quantize(cudaStream_t, TensorView x, RowScaledView quantized, TensorView weight,
                         TensorView bias, float eps, const NormFusion & = {},
                         TensorView retained = {});
void positional_add(cudaStream_t, TensorView x, TensorView y, TensorView table, int sequence_length,
                    int offset = 0, TensorView alpha = {},
                    const void *const *source_rows = nullptr,
                    void *const *retained_rows = nullptr);
// Fused projection epilogue; the rotary prefix can span the whole projection or
// be applied independently to each head. Values receive only their bias.
void qkv_rotary(cudaStream_t, TensorView qkv, TensorView bias, TensorView rotary,
                int sequence_length, int heads, int head_dim, int rotary_dim, bool per_head,
                int offset = 0);
void global_response_norm(cudaStream_t, TensorView x, TensorView y, TensorView gamma,
                          TensorView beta, int batch, int time, TensorView statistics);
void gaussian_noise(cudaStream_t, TensorView out, int time, const int *prefix_lengths,
                    const uint64_t *seeds, const uint64_t *subsequences, float temperature);
struct EulerOutput {
    TensorView values;
    int offset = 0, frames = 0;
    float minimum = -1, maximum = 1;
    // Prompt context and generated history are handled in the integration pass.
    const void *const *prompt_rows = nullptr;
    void *const *history_rows = nullptr;
    void *const *value_rows = nullptr; // Optional denormalized scatter, in the same pass.
};
// Optional final-step output performs the affine range conversion while the
// integrated value is in registers, excluding a prefix without gathering.
void euler_step(cudaStream_t, TensorView x, TensorView velocity, TensorView negative, int time,
                const int *prefix_lengths, float delta, float cfg_scale, const EulerOutput & = {});
struct EmbeddingFusion {
    TensorView residual, residual_default, positions, position_alpha, segment, norm_weight, norm_bias;
    int sequence_length = 1, position_offset = 0, segment_id = 0;
    int batch = 0, batch_stride_rows = 0;
    int input_repeat = 1;
    float table_scale = 1, epsilon = 1.e-5f;
    // Optional device indices for packed output, per-token positions and a
    // shared conditioning projection. Negative indices use residual_default
    // (a channel vector), or contribute zero when that vector is absent.
    const int *output_rows = nullptr, *position_ids = nullptr, *residual_rows = nullptr;
    int input_rows = 0;
    const int *const *id_rows = nullptr;
    // Optional borrowed rows override the dense residual for those tokens.
    // Null entries continue to use residual_rows; no gather buffer is needed.
    const void *const *residual_sources = nullptr;
};
void embedding(cudaStream_t, TensorView table, const int *ids, TensorView out,
               const EmbeddingFusion & = {});
void temporal_norm(cudaStream_t, TensorView x, TensorView y, TensorView weight, TensorView bias,
                   int batch, int time, float epsilon, Activation, TensorView statistics);
void gated(cudaStream_t, TensorView input, TensorView out, Activation left, int time,
           TensorView condition = {}, TensorView residual = {}, const SequenceLayout *packed = nullptr,
           const void *const *condition_sources = nullptr, int condition_offset = 0);
void temporal_mean(cudaStream_t, TensorView x, TensorView out, int time);
void residual_split(cudaStream_t, TensorView projected, TensorView residual, TensorView skip,
                    bool first, bool last, int time = 0, int valid_time = 0,
                    const SequenceLayout *packed = nullptr);
} // namespace tts::ops
