#pragma once
#include "tensor.h"
#include <cuda_runtime_api.h>
namespace tts {
void audio_embedding(cudaStream_t stream, TensorView table, const int *codes, TensorView out,
                     int batch, int branches, int codebooks, int codebook = -1);
void rvq(cudaStream_t stream, TensorView embeddings, const int *codes, TensorView out, int batch,
         int first, int count, int codebooks);
void snake(cudaStream_t stream, TensorView x, TensorView alpha, TensorView beta, TensorView out);
void conv_accumulate(cudaStream_t stream, TensorView projected, TensorView pending, TensorView bias,
                     TensorView out, int batch, int time, int kernel, int dilation, int stride,
                     bool transpose, const int *frame);
void depthwise(cudaStream_t stream, TensorView x, TensorView history, TensorView w, TensorView bias,
               TensorView out, int batch, int time, int kernel, const int *frame);
void pcm(cudaStream_t stream, TensorView x, int16_t *out);
} // namespace tts
