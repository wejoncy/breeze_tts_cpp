#include "encoder_kernels.h"
#include <cmath>
#include <type_traits>
namespace tts {
template <bool PowerOfTwo>
__global__ void first_conv(const float *x, const float *w, const float *bias, float *out, int time,
                           int channels, int kernel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= time * channels)
        return;
    int t = PowerOfTwo ? i >> (__ffs(channels) - 1) : i / channels,
        c = PowerOfTwo ? i & (channels - 1) : i % channels;
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
    if (x.cols != 1 || out.cols < 1 || out.rows != x.rows)
        throw std::invalid_argument("encoder first convolution shape");
    auto launch = [&](auto pow2) {
        first_conv<decltype(pow2)::value><<<(out.numel() + 255) / 256, 256, 0, stream>>>(
            x.ptr<float>(), w.ptr<float>(), b.ptr<float>(), out.ptr<float>(), x.rows, out.cols,
            kernel);
    };
    if ((out.cols & (out.cols - 1)) == 0)
        launch(std::true_type{});
    else
        launch(std::false_type{});
}
template <int Kernel, int Stride, bool Replicate>
__global__ void conv_finish(const float *x, const float *p, const float *w, const float *bias,
                            float *out, const float *residual, int in_time, int in_channels,
                            int out_time, int channels, bool elu, int dynamic_kernel,
                            int dynamic_stride, int dilation, bool replicate) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_time * channels)
        return;
    int t = Kernel ? i >> (__ffs(channels) - 1) : i / channels,
        c = Kernel ? i & (channels - 1) : i % channels;
    const int kernel = Kernel ? Kernel : dynamic_kernel;
    const int stride = Stride ? Stride : dynamic_stride;
    const int taps = kernel / stride;
    int width = kernel * channels;
    float sum = bias ? bias[c] : 0;
#pragma unroll
    for (int k = 0; k < kernel; k++) {
        int src = t * stride - (kernel - 1) * (Kernel ? 1 : dilation) + stride - 1 +
                  k * (Kernel ? 1 : dilation);
        if (src >= 0 && src < in_time) {
            int phase = k % stride, j = Kernel ? t - taps + 1 + k / stride : src / stride;
            sum += p[size_t(j) * width + (phase * taps + k / stride) * channels + c];
        } else if (Kernel ? Replicate : replicate) {
            int phase = k % stride;
            const float *wp = w + size_t((phase * taps + k / stride) * channels + c) * in_channels;
            for (int d = 0; d < in_channels; d++)
                sum += x[size_t(src < 0 ? 0 : in_time - 1) * in_channels + d] * wp[d];
        }
    }
    if (residual)
        sum += residual[i];
    out[i] = elu && sum < 0 ? expm1f(sum) : sum;
}
void encoder_conv_finish(cudaStream_t stream, TensorView x, TensorView p, TensorView w,
                         TensorView b, TensorView out, int kernel, int stride, bool replicate,
                         bool elu, TensorView residual, int dilation) {
    // Shape specializations preserve unrolled arithmetic; other valid geometries
    // use the same fused accumulation, bias, residual and activation operation.
#define RUN(K, S, R)                                                                               \
    conv_finish<K, S, R><<<(out.numel() + 255) / 256, 256, 0, stream>>>(                           \
        x.ptr<float>(), p.ptr<float>(), w.ptr<float>(), b.ptr<float>(), out.ptr<float>(),          \
        residual.ptr<float>(), x.rows, x.cols, out.rows, out.cols, elu, kernel, stride, dilation,  \
        replicate)
    if (out.cols < 1 || kernel < 1 || stride < 1 || kernel % stride || dilation < 1 ||
        (stride > 1 && dilation > 1))
        throw std::invalid_argument("invalid encoder convolution geometry");
    if (dilation != 1 || (out.cols & (out.cols - 1)) ||
        (replicate && !(stride == 2 && kernel == 4))) {
        RUN(0, 0, false);
    } else if (stride == 1 && kernel == 1) {
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
    } else {
        RUN(0, 0, false);
    }
#undef RUN
}
// Scores = residual * codebook^T. The norm and -2*x.e term suffice for argmin;
// the common ||x||^2 term is deliberately omitted. Select and residual subtraction
// share one CTA, so no standalone gather/copy or distance tensor is required.
template <int Entries = 0, int Dimension = 0>
__global__ void quantize(float *x, const float *scores, const float *table, const float *norms,
                         int *codes, int codebook, int codebooks, int entries, int dimension) {
    if constexpr (Entries)
        entries = Entries;
    if constexpr (Dimension)
        dimension = Dimension;
    int row = blockIdx.x, lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    float best = INFINITY;
    int index = 0;
    for (int k = threadIdx.x; k < entries; k += 256) {
        float d = norms[k] - 2.f * scores[size_t(row) * entries + k];
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
        codes[size_t(row) * codebooks + codebook] = index;
    }
    __syncthreads();
    for (int d = threadIdx.x; d < dimension; d += blockDim.x)
        x[size_t(row) * dimension + d] -= table[size_t(chosen) * dimension + d];
}
void encoder_quantize(cudaStream_t stream, TensorView x, TensorView scores, TensorView table,
                      TensorView norms, int *codes, int cb, int codebooks) {
    if (cb < 0 || cb >= codebooks || scores.cols != table.rows || x.cols != table.cols ||
        x.rows != scores.rows || norms.numel() != size_t(table.rows))
        throw std::invalid_argument("encoder quantizer shape");
    auto launch = [&](auto entries, auto dimension) {
        quantize<decltype(entries)::value, decltype(dimension)::value><<<x.rows, 256, 0, stream>>>(
            x.ptr<float>(), scores.ptr<float>(), table.ptr<float>(), norms.ptr<float>(), codes, cb,
            codebooks, table.rows, table.cols);
    };
    if (table.rows == 2048 && table.cols == 256)
        launch(std::integral_constant<int, 2048>{}, std::integral_constant<int, 256>{});
    else
        launch(std::integral_constant<int, 0>{}, std::integral_constant<int, 0>{});
}
} // namespace tts
