#pragma once
#include "ops/sequence.h"

namespace tts::ops {
// CudaContext launch interface. Occupancy is read once at initialization.
int normalized_linear_capacity(int multiprocessors, int latency_warps, int &writeback_capacity);
bool normalized_linear(cudaStream_t, TensorView input, TensorView normalized,
                       TensorView norm_weight, TensorView norm_bias, float epsilon,
                       const NormFusion &, TensorView weight, TensorView output,
                       const Epilogue &, int resident_blocks,
                       const QkvProjectionOutput *writeback = nullptr);
// Same row statistics and rounding as normalized_linear; occupancy and the
// existing prepared GEMV family policy are supplied during CUDA initialization.
int normalized_quantized_linear_capacity(Type format, int multiprocessors);
bool normalized_quantized_linear(cudaStream_t, TensorView input, TensorView normalized,
                                TensorView norm_weight, TensorView norm_bias, float epsilon,
                                const NormFusion &, RowScaledView weight, TensorView output,
                                const Epilogue &, int resident_blocks,
                                RowScaledView encoded = {},
                                const QkvProjectionOutput *writeback = nullptr);
} // namespace tts::ops
