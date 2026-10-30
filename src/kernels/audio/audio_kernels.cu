#include "kernels/audio/audio_kernels.h"
#include "kernels/common/cuda_math.cuh"
#include <cmath>
namespace tts {
__global__ void audio_embed_kernel(const bf16 *w, const int *codes, bf16 *out, int batch,
                                   int branches, int cols, int cb, int stride, int codebooks,
                                   int vocab, const int *const *code_rows, const int *row_request) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(batch) * branches * cols)
        return;
    int row = i / cols, b = row % batch, c = i % cols;
    int owner = row_request ? row_request[row] : b;
    codes = code_rows ? (owner >= 0 ? code_rows[owner] : nullptr) : codes + b * codebooks;
    if (!codes) {
        out[size_t(row) * stride + c] = r<bf16>(0.f);
        return;
    }
    float sum = 0;
    for (int k = cb < 0 ? 0 : cb; k < (cb < 0 ? codebooks : cb + 1); k++)
        sum += f(w[(size_t(k) * vocab + codes[k]) * cols + c]);
    out[size_t(row) * stride + c] = r<bf16>(sum);
}
void audio_embedding(cudaStream_t stream, TensorView table, const int *codes, TensorView out,
                     int batch, int branches, int codebooks, int cb, const int *const *code_rows,
                     const int *row_request) {
    size_t n = size_t(out.rows) * out.cols;
    audio_embed_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        table.ptr<bf16>(), codes, out.ptr<bf16>(), batch, branches, out.cols, cb, out.ld(),
        codebooks, table.rows / codebooks, code_rows, row_request);
}

__global__ void rvq_kernel(const float *e, const int *codes, float *out, int batch, int first,
                           int count, int dim, int codebooks, int entries,
                           const int *const *code_rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch * dim)
        return;
    int b = i / dim, d = i % dim;
    codes = code_rows ? code_rows[b] : codes + b * codebooks;
    if (!codes) {
        out[i] = 0.f;
        return;
    }
    float sum = 0;
    for (int k = first; k < first + count; k++)
        sum += e[(size_t(k - first) * entries + codes[k]) * dim + d];
    out[i] = sum;
}
void rvq(cudaStream_t stream, TensorView e, const int *codes, TensorView out, int batch, int first,
         int count, int codebooks, const int *const *code_rows) {
    int n = batch * out.cols;
    rvq_kernel<<<(n + 255) / 256, 256, 0, stream>>>(e.ptr<float>(), codes, out.ptr<float>(), batch,
                                                    first, count, out.cols, codebooks,
                                                    e.rows / count, code_rows);
}
__global__ void snake_kernel(const float *x, const float *a, const float *b, float *y, int n,
                             int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float s = sinf(x[i] * expf(a[i % cols]));
        y[i] = x[i] + s * s / (expf(b[i % cols]) + 1e-9f);
    }
}
void snake(cudaStream_t stream, TensorView x, TensorView a, TensorView b, TensorView out) {
    int n = x.rows * x.cols;
    snake_kernel<<<(n + 255) / 256, 256, 0, stream>>>(x.ptr<float>(), a.ptr<float>(),
                                                      b.ptr<float>(), out.ptr<float>(), n, x.cols);
}

// Streaming overlap accumulation. GEMM evaluates every tap on new inputs
// exactly once. This kernel sums actual convolution contributions, applies
// bias, and commits future output state; it never rearranges/copies inputs.
__device__ float conv_output(const float *projected, const float *pending, int batch, int t,
                             int channel, int time, int channels, int kernel, int dilation,
                             int stride, bool trans, int horizon, int capacity, int frame) {
    int position = (frame * time * (trans ? stride : 1) + t) % capacity;
    float sum = t < horizon ? pending[size_t(position) * channels + channel] : 0.f;
    for (int k = 0; k < kernel; k++) {
        int src = trans ? t - k : t - (kernel - 1 - k) * dilation;
        if (trans) {
            if (src % stride)
                continue;
            src /= stride;
        }
        if (src >= 0 && src < time)
            sum += projected[((size_t(batch) * time + src) * kernel + k) * channels + channel];
    }
    return sum;
}
__global__ void conv_accumulate_kernel(const float *projected, float *pending, const float *bias,
                                       float *out, int batch, int time, int channels, int kernel,
                                       int dilation, int stride, bool trans, int horizon,
                                       int capacity, const int *frames, const ConvHistory *states) {
    int outtime = time * (trans ? stride : 1), i = blockIdx.x * blockDim.x + threadIdx.x;
    if (outtime >= horizon) {
        if (i >= batch * capacity * channels)
            return;
        int c = i % channels, t = (i / channels) % capacity, b = i / channels / capacity;
        if (states && !states[b].frame) {
            if (t < outtime)
                out[(size_t(b) * outtime + t) * channels + c] = 0.f;
            return;
        }
        int frame = states ? *states[b].frame : frames[b];
        pending = states ? states[b].values : pending + size_t(b) * capacity * channels;
        float value = conv_output(projected, pending, b, t, c, time, channels, kernel, dilation,
                                  stride, trans, horizon, capacity, frame);
        if (t < outtime)
            out[(size_t(b) * outtime + t) * channels + c] = value + (bias ? bias[c] : 0.f);
        else
            pending[((frame * outtime + t) % capacity) * channels + c] = value;
    } else {
        // If a chunk is shorter than its future-state horizon, one thread
        // owns each channel and visits time in order to avoid state hazards.
        if (i >= batch * channels)
            return;
        int b = i / channels, c = i % channels;
        if (states && !states[b].frame) {
            for (int t = 0; t < outtime; ++t)
                out[(size_t(b) * outtime + t) * channels + c] = 0.f;
            return;
        }
        int frame = states ? *states[b].frame : frames[b];
        pending = states ? states[b].values : pending + size_t(b) * capacity * channels;
        for (int t = 0; t < capacity; t++) {
            float value = conv_output(projected, pending, b, t, c, time, channels, kernel, dilation,
                                      stride, trans, horizon, capacity, frame);
            if (t < outtime)
                out[(size_t(b) * outtime + t) * channels + c] = value + (bias ? bias[c] : 0.f);
            else
                pending[((frame * outtime + t) % capacity) * channels + c] = value;
        }
    }
}
void conv_accumulate(cudaStream_t stream, TensorView projected, TensorView pending, TensorView bias,
                     TensorView out, int batch, int time, int kernel, int dilation, int stride,
                     bool trans, const int *frames, const ConvHistory *states) {
    int horizon = trans ? kernel - stride : (kernel - 1) * dilation,
        outtime = time * (trans ? stride : 1), capacity = outtime + horizon;
    int n = batch * out.cols * (outtime >= horizon ? capacity : 1);
    conv_accumulate_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        projected.ptr<float>(), pending.ptr<float>(), bias.ptr<float>(), out.ptr<float>(), batch,
        time, out.cols, kernel, dilation, stride, trans, horizon, capacity, frames, states);
}
// One thread owns a channel across the whole chunk. It reads all old history
// before committing new samples, so no other CTA can race an overwritten slot.
__global__ void depthwise_kernel(const float *x, float *hist, const float *w, const float *bias,
                                 float *y, int batch, int time, int channels, int kernel,
                                 const int *frame, const ConvHistory *states) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch * channels)
        return;
    int c = i % channels, b = i / channels, length = kernel - 1;
    if (states && !states[b].frame) {
        for (int t = 0; t < time; ++t)
            y[(size_t(b) * time + t) * channels + c] = 0.f;
        return;
    }
    int position = states ? *states[b].frame : frame[b];
    hist = states ? states[b].values : hist + size_t(b) * length * channels;
    for (int t = 0; t < time; ++t) {
        float sum = bias ? bias[c] : 0;
        for (int k = 0; k < kernel; ++k) {
            int s = t - kernel + 1 + k;
            float v =
                s >= 0 ? x[(size_t(b) * time + s) * channels + c]
                       : hist[(((position * time + s) % length + length) % length) * channels + c];
            sum += v * w[c * kernel + k];
        }
        y[(size_t(b) * time + t) * channels + c] = sum;
    }
    for (int t = max(0, time - length); t < time; ++t)
        hist[((position * time + t) % length) * channels + c] =
            x[(size_t(b) * time + t) * channels + c];
}
void depthwise(cudaStream_t stream, TensorView x, TensorView hist, TensorView w, TensorView bias,
               TensorView out, int batch, int time, int kernel, const int *frame,
               const ConvHistory *states) {
    int n = batch * x.cols;
    depthwise_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        x.ptr<float>(), hist.ptr<float>(), w.ptr<float>(), bias.ptr<float>(), out.ptr<float>(),
        batch, time, x.cols, kernel, frame, states);
}
__global__ void pcm_kernel(const float *x, int16_t *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = int16_t(lrintf(fminf(1.f, fmaxf(-1.f, x[i])) * 32767.f));
}
void pcm(cudaStream_t stream, TensorView x, int16_t *out) {
    int n = x.rows * x.cols;
    pcm_kernel<<<(n + 255) / 256, 256, 0, stream>>>(x.ptr<float>(), out, n);
}
} // namespace tts
