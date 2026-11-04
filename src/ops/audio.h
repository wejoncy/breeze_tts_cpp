#pragma once
#include "core/tensor.h"
namespace tts {
struct ConvHistory {
    float *values;
    const int *frame;
};
void audio_embedding( TensorView table, const int *codes, TensorView out,
                     int batch, int branches, int codebooks, int codebook = -1,
                     const int *const *code_rows = nullptr, const int *row_request = nullptr);
void rvq( TensorView embeddings, const int *codes, TensorView out, int batch,
         int first, int count, int codebooks, const int *const *code_rows = nullptr);
void snake( TensorView x, TensorView alpha, TensorView beta, TensorView out);
void conv_accumulate( TensorView projected, TensorView pending, TensorView bias,
                     TensorView out, int batch, int time, int kernel, int dilation, int stride,
                     bool transpose, const int *frame, const ConvHistory *states = nullptr);
void depthwise( TensorView x, TensorView history, TensorView w, TensorView bias,
               TensorView out, int batch, int time, int kernel, const int *frame,
               const ConvHistory *states = nullptr);
void pcm( TensorView x, int16_t *out);
} // namespace tts
