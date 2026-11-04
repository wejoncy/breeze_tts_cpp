#include "runtime/cuda_context.h"
#include "ops/vector_quantization.h"
#include "ops/kernels/common/cuda_math.cuh"
#include "runtime/device.h"
#include <cuda_fp16.h>

namespace tts {
template <class T>
__global__ void quantize(T *x, const T *scores, const T *table, const T *norms,
                         int *codes, int codebook, int codebooks, int entries, int dimension,
                         int xs, int ss, int ts, bool subtract, bool expanded) {
    int row = blockIdx.x, lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    __shared__ float vals[8], input_norm;
    __shared__ int ids[8], chosen;
    if (expanded) {
        float norm = 0;
        for (int d = threadIdx.x; d < dimension; d += blockDim.x) {
            float v = f(x[size_t(row) * xs + d]);
            norm += rounded<T>(v * v);
        }
        norm = warp_sum(norm);
        if (!lane) vals[warp] = norm;
        __syncthreads();
        if (!threadIdx.x) {
            norm = 0;
            for (int w = 0; w < 8; ++w) norm += vals[w];
            input_norm = rounded<T>(norm);
        }
        __syncthreads();
    }
    float best = INFINITY;
    int index = 0;
    for (int k = threadIdx.x; k < entries; k += blockDim.x) {
        float score = f(scores[size_t(row) * ss + k]);
        float distance = expanded ? rounded<T>(rounded<T>(input_norm - score) + f(norms[k]))
                                  : f(norms[k]) - 2.f * score;
        if (distance < best || (distance == best && k < index)) {
            best = distance; index = k;
        }
    }
    for (int shift = 16; shift; shift /= 2) {
        float v = __shfl_down_sync(0xffffffff, best, shift);
        int k = __shfl_down_sync(0xffffffff, index, shift);
        if (v < best || (v == best && k < index)) { best = v; index = k; }
    }
    if (!lane) { vals[warp] = best; ids[warp] = index; }
    __syncthreads();
    if (!threadIdx.x) {
        for (int w = 1; w < 8; ++w)
            if (vals[w] < best || (vals[w] == best && ids[w] < index)) {
                best = vals[w]; index = ids[w];
            }
        chosen = index;
        codes[size_t(row) * codebooks + codebook] = index;
    }
    if (subtract) {
        __syncthreads();
        for (int d = threadIdx.x; d < dimension; d += blockDim.x)
            x[size_t(row) * xs + d] = r<T>(f(x[size_t(row) * xs + d]) -
                                           f(table[size_t(chosen) * ts + d]));
    }
}
void vector_quantize( TensorView x, TensorView scores, TensorView table,
                      TensorView norms, int *codes, int cb, int codebooks, bool subtract, bool expanded) {
    const auto stream = execution_context(x).stream;
    if (cb < 0 || cb >= codebooks || !codes || !x.data || !scores.data || !table.data ||
        !norms.data || scores.cols != table.rows || x.cols != table.cols ||
        x.rows != scores.rows || norms.numel() != size_t(table.rows) || !norms.dense() ||
        x.type != scores.type || x.type != table.type || x.type != norms.type ||
        (x.type != Type::BF16 && x.type != Type::F16 && x.type != Type::F32))
        throw std::invalid_argument("vector quantizer geometry/dtype mismatch");
    auto launch = [&](auto value) {
        using T = decltype(value);
        quantize<<<x.rows, 256, 0, stream>>>(x.ptr<T>(), scores.ptr<T>(), table.ptr<T>(), norms.ptr<T>(),
            codes, cb, codebooks, table.rows, table.cols, x.ld(), scores.ld(), table.ld(), subtract, expanded);
    };
    if (x.type == Type::BF16) launch(bf16{});
    else if (x.type == Type::F16) launch(__half{});
    else launch(float{});
    CUDA(cudaGetLastError());
}
} // namespace tts
