#include "ops/cuda/backend.h"
#include "ops/spectral.h"
#include "runtime/device.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

namespace tts {
__global__ void frame_window(const float *waveform, const float *window, float *out,
                             int rows, int fft, int samples, int time, int hop, int padding,
                             int actual_rows) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(rows) * fft) return;
    int row = i / fft, d = i % fft;
    if (row >= actual_rows) { out[i] = 0; return; }
    int source = (row % time) * hop - padding + d;
    // Reflect excludes the boundary sample, matching STFT's reflect padding.
    int period = 2 * (samples - 1);
    source %= period;
    if (source < 0) source += period;
    if (source >= samples) source = period - source;
    out[i] = waveform[size_t(row / time) * samples + source] * window[d];
}
void stft_window( TensorView waveform, TensorView window, TensorView frames,
                 int batch, int samples, int time, int hop, int padding) {
    const auto stream = execution_context(waveform).stream;
    if (batch < 1 || samples < 2 || samples > INT32_MAX / 2 || time < 1 || hop < 1 || padding < 0 ||
        waveform.rows != int64_t(batch) * samples || waveform.cols != 1 || !waveform.dense() ||
        waveform.type != Type::F32 || window.type != Type::F32 || window.rows != 1 ||
        !window.dense() || frames.type != Type::F32 || !frames.dense() ||
        frames.rows < int64_t(batch) * time || frames.cols != window.cols)
        throw std::invalid_argument("invalid STFT frame geometry");
    frame_window<<<(frames.numel() + 255) / 256, 256, 0, stream>>>(waveform.ptr<float>(),
        window.ptr<float>(), frames.ptr<float>(), frames.rows, frames.cols, samples, time, hop,
        padding, batch * time);
    CUDA(cudaGetLastError());
}
template <class T>
__global__ void magnitude(const float2 *spectrum, T *out, int rows, int cols, int stride, float eps) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(rows) * cols) return;
    float2 value = spectrum[i];
    out[(i / cols) * stride + i % cols] = T(sqrtf(value.x * value.x + value.y * value.y + eps));
}
template <class T>
__global__ void logarithm(const float *x, T *y, int rows, int cols, int xs, int ys,
                           float floor, float scale, float bias) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(rows) * cols) return;
    int row = i / cols, col = i % cols;
    y[size_t(row) * ys + col] = T(logf(fmaxf(x[size_t(row) * xs + col], floor)) * scale + bias);
}
static void check_feature(TensorView y) {
    if (!y.data || (y.type != Type::BF16 && y.type != Type::F16 && y.type != Type::F32))
        throw std::invalid_argument("invalid spectral output dtype");
}
void spectrum_magnitude( const void *spectrum, TensorView out, float epsilon) {
    const auto stream = execution_context(out).stream;
    check_feature(out);
    if (!spectrum || epsilon < 0 || !std::isfinite(epsilon))
        throw std::invalid_argument("invalid spectral magnitude");
    auto launch = [&](auto value) {
        magnitude<<<(out.numel() + 255) / 256, 256, 0, stream>>>(static_cast<const float2 *>(spectrum),
            out.ptr<decltype(value)>(), out.rows, out.cols, out.ld(), epsilon);
    };
    if (out.type == Type::F16) launch(__half{});
    else if (out.type == Type::BF16) launch(__nv_bfloat16{});
    else launch(float{});
    CUDA(cudaGetLastError());
}
void spectral_log( TensorView input, TensorView out, float floor,
                  float scale, float bias) {
    const auto stream = execution_context(input).stream;
    check_feature(out);
    if (input.type != Type::F32 || input.rows != out.rows || input.cols != out.cols ||
        !(floor > 0) || !std::isfinite(floor) || !std::isfinite(scale) || !std::isfinite(bias))
        throw std::invalid_argument("invalid spectral log transform");
    auto launch = [&](auto value) {
        logarithm<<<(out.numel() + 255) / 256, 256, 0, stream>>>(input.ptr<float>(),
            out.ptr<decltype(value)>(), out.rows, out.cols, input.ld(), out.ld(), floor, scale, bias);
    };
    if (out.type == Type::F16) launch(__half{});
    else if (out.type == Type::BF16) launch(__nv_bfloat16{});
    else launch(float{});
    CUDA(cudaGetLastError());
}
} // namespace tts
