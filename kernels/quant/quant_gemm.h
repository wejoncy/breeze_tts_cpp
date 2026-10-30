#pragma once
#include <cuda_runtime.h>
#include <cstddef>
namespace tts {
// Hopper-native QGMMA E4M3. Single-partition paths apply both row scales in
// the GEMM epilogue. Split paths use strided K views and reduce FP32 partials.
// Storage is backend-owned; no input copies, weight repacking, or allocation.
void fp8_hopper_gemm(void *x, const float *sx, void *w, const float *sw, void *y, int rows,
                     int output_channels, int input_channels, void *workspace,
                     size_t workspace_bytes, cudaStream_t stream, int tile_channels,
                     int tile_tokens, int splits);
void int8_hopper_gemm(void *x, const float *sx, void *w, const float *sw, void *y, int rows,
                      int output_channels, int input_channels, void *workspace,
                      size_t workspace_bytes, cudaStream_t stream, int tile_channels,
                      int tile_tokens, int splits);
void int8_sm80_gemm(void *x, const float *sx, void *w, const float *sw, void *y, int rows,
                    int output_channels, int input_channels, void *workspace,
                    size_t workspace_bytes, cudaStream_t stream, int tile_channels, int tile_tokens,
                    int splits);
} // namespace tts
