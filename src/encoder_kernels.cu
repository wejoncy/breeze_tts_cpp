#include "encoder_kernels.h"
#include <cmath>
namespace tts {
__global__ void first_conv(const float *x, const float *w, const float *bias, float *out, int time,
                           int channels, int kernel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= time * channels)
        return;
    int t = i >> (__ffs(channels) - 1), c = i & (channels - 1);
    float sum = bias ? bias[c] : 0;
    for (int k = 0; k < kernel; k++) {
        int s = t - kernel + 1 + k;
        if (s >= 0)
            sum += x[s] * w[k * channels + c];
    }
    out[i] = sum;
}
void encoder_first_conv(cudaStream_t stream, TensorView x, TensorView w, TensorView b, TensorView out,
                        int kernel) {
    first_conv<<<(out.numel() + 255) / 256, 256, 0, stream>>>(
        x.ptr<float>(), w.ptr<float>(), b.ptr<float>(), out.ptr<float>(), x.rows, out.cols, kernel);
}
template <int Kernel, int Stride, bool Replicate>
__global__ void conv_finish(const float *x, const float *p, const float *w, const float *bias,
                            float *out, const float *residual, int in_time, int in_channels,
                            int out_time, int channels, bool elu) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_time * channels)
        return;
    int t = i >> (__ffs(channels) - 1), c = i & (channels - 1);
    constexpr int taps = Kernel / Stride;
    int width = Kernel * channels;
    float sum = bias ? bias[c] : 0;
#pragma unroll
    for (int k = 0; k < Kernel; k++) {
        int src = t * Stride - (Kernel - Stride) + k;
        if (src >= 0 && src < in_time) {
            int phase = k % Stride, j = t - taps + 1 + k / Stride;
            sum += p[size_t(j) * width + (phase * taps + k / Stride) * channels + c];
        } else if (Replicate) {
            int phase = k % Stride;
            const float *wp = w + size_t((phase * taps + k / Stride) * channels + c) * in_channels;
            for (int d = 0; d < in_channels; d++)
                sum += x[size_t(src < 0 ? 0 : in_time - 1) * in_channels + d] * wp[d];
        }
    }
    if (residual)
        sum += residual[i];
    out[i] = elu && sum < 0 ? expm1f(sum) : sum;
}
void encoder_conv_finish(cudaStream_t stream, TensorView x, TensorView p, TensorView w, TensorView b,
                         TensorView out, int kernel, int stride, bool replicate, bool elu,
                         TensorView residual) {
    if (out.cols < 1 || (out.cols & (out.cols - 1)))
        throw std::invalid_argument("encoder channels must be a power of two");
    // Fixed checkpoint geometries eliminate integer division/remainder in each
    // tap. GEMMs already produce phase/tap-major columns; no layout kernel.
#define RUN(K, S, R)                                                                               \
    conv_finish<K, S, R><<<(out.numel() + 255) / 256, 256, 0, stream>>>(                         \
        x.ptr<float>(), p.ptr<float>(), w.ptr<float>(), b.ptr<float>(), out.ptr<float>(),          \
        residual.ptr<float>(), x.rows, x.cols, out.rows, out.cols, elu)
    if (stride == 1 && kernel == 1) {
        RUN(1, 1, false);
    } else if (stride == 1 && kernel == 3) {
        RUN(3, 1, false);
    } else if (stride == 4 && kernel == 8) {
        RUN(8, 4, false);
    } else if (stride == 5 && kernel == 10) {
        RUN(10, 5, false);
    } else if (stride == 6 && kernel == 12) {
        RUN(12, 6, false);
    } else if (stride == 8 && kernel == 16) {
        RUN(16, 8, false);
    } else if (stride == 2 && kernel == 4 && replicate) {
        RUN(4, 2, true);
    } else
        throw std::invalid_argument("unsupported encoder convolution geometry");
#undef RUN
}
// Scores = residual * codebook^T. The norm and -2*x.e term suffice for argmin;
// the common ||x||^2 term is deliberately omitted. Select and residual subtraction
// share one CTA, so no standalone gather/copy or distance tensor is required.
__global__ void quantize(float *x, const float *scores, const float *table, const float *norms,
                         int *codes, int codebook, int rows) {
    int row = blockIdx.x, lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    float best = INFINITY;
    int index = 0;
    for (int k = threadIdx.x; k < 2048; k += 256) {
        float d = norms[k] - 2.f * scores[row * 2048 + k];
        if (d < best || (d == best && k < index)) {
            best = d;
            index = k;
        }
    }
    for (int shift = 16; shift; shift /= 2) {
        float v = __shfl_down_sync(0xffffffff, best, shift);
        int k = __shfl_down_sync(0xffffffff, index, shift);
        if (v < best || (v == best && k < index)) {
            best = v;
            index = k;
        }
    }
    __shared__ float vals[8];
    __shared__ int ids[8], chosen;
    if (!lane) {
        vals[warp] = best;
        ids[warp] = index;
    }
    __syncthreads();
    if (!threadIdx.x) {
        for (int w = 1; w < 8; w++)
            if (vals[w] < best || (vals[w] == best && ids[w] < index)) {
                best = vals[w];
                index = ids[w];
            }
        chosen = index;
        codes[row * 16 + codebook] = index;
    }
    __syncthreads();
    int d = threadIdx.x;
    x[row * 256 + d] -= table[chosen * 256 + d];
}
void encoder_quantize(cudaStream_t stream, TensorView x, TensorView scores, TensorView table,
                      TensorView norms, int *codes, int cb) {
    quantize<<<x.rows, 256, 0, stream>>>(x.ptr<float>(), scores.ptr<float>(), table.ptr<float>(),
                                           norms.ptr<float>(), codes, cb, x.rows);
}
} // namespace tts
