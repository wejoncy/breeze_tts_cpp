#include "sampling.h"
#include "cuda_math.cuh"
#include <cmath>
#include <cub/block/block_radix_sort.cuh>
namespace tts {
__device__ uint64_t splitmix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
template <class T>
__device__ float guided_logit(const T *logits, int vocab, int b, int batch, int branches, int cb,
                              int id, float cfg, float cfg_ref, float cfg_ins, float penalty,
                              const int *seen) {
    if (!(id < 2048 || (cb == 0 && id == 2051)))
        return -INFINITY;
    float v = f(logits[b * vocab + id]);
    if (branches == 2) {
        float u = f(logits[(batch + b) * vocab + id]);
        v = u + cfg * (v - u);
    }
    if (branches == 3) {
        float r = f(logits[(batch + b) * vocab + id]),
              ins = f(logits[(2 * batch + b) * vocab + id]);
        v = v + cfg_ref * (r - v) + cfg_ins * (ins - v);
    }
    if (cb == 0 && id < 2048 && seen && seen[b * 2048 + id])
        v = v < 0 ? v * penalty : v / penalty;
    return v;
}
template <class T>
__global__ void greedy_kernel(const T *logits, int vocab, int *codes, int batch, int branches,
                              int cb, float cfg, int *done, float cfg_ref, float cfg_ins,
                              float penalty, int *seen) {
    __shared__ float vals[8];
    __shared__ int ids[8];
    int b = blockIdx.x, id = 0;
    float val = -INFINITY;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        float v = guided_logit(logits, vocab, b, batch, branches, cb, i, cfg, cfg_ref, cfg_ins,
                               penalty, seen);
        if (v > val || (v == val && i < id)) {
            val = v;
            id = i;
        }
    }
    for (int d = 16; d; d /= 2) {
        float v = __shfl_down_sync(0xffffffff, val, d);
        int i = __shfl_down_sync(0xffffffff, id, d);
        if (v > val || (v == val && i < id)) {
            val = v;
            id = i;
        }
    }
    if (threadIdx.x % 32 == 0) {
        vals[threadIdx.x / 32] = val;
        ids[threadIdx.x / 32] = id;
    }
    __syncthreads();
    if (!threadIdx.x) {
        for (int j = 0; j < 8; j++)
            if (vals[j] > val || (vals[j] == val && ids[j] < id)) {
                val = vals[j];
                id = ids[j];
            }
        if (cb == 0 && id == 2051) {
            done[b] = 1;
            id = 0;
        }
        if (cb == 0 && !done[b] && seen)
            seen[b * 2048 + id] = 1;
        codes[b * 16 + cb] = id;
    }
}
template <class T>
__global__ void sample_kernel(const T *logits, int vocab, int *codes, uint64_t *rng, int batch,
                              int branches, int cb, float cfg, float temp, int topk, int *done,
                              float topp, float cfg_ref, float cfg_ins, float penalty, int *seen) {
    using Sort = cub::BlockRadixSort<float, 256, 16, int>;
    __shared__ union {
        typename Sort::TempStorage sort;
        struct {
            float values[4096];
            int ids[4096];
        } ordered;
    } storage;
    int b = blockIdx.x;
    float values[16];
    int ids[16];
#pragma unroll
    for (int j = 0; j < 16; j++) {
        int id = threadIdx.x * 16 + j;
        ids[j] = id;
        values[j] = guided_logit(logits, vocab, b, batch, branches, cb, id, cfg, cfg_ref, cfg_ins,
                                 penalty, seen);
    }
    Sort(storage.sort).SortDescending(values, ids);
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 16; j++) {
        storage.ordered.values[threadIdx.x * 16 + j] = values[j];
        storage.ordered.ids[threadIdx.x * 16 + j] = ids[j];
    }
    __syncthreads();
    if (!threadIdx.x) {
        auto *v = storage.ordered.values;
        auto *id = storage.ordered.ids;
        int valid = cb == 0 ? 2049 : 2048, k = topk > 0 ? min(topk, valid) : valid;
        // HF top-k keeps all logits tied at the threshold.
        while (k < valid && v[k] == v[k - 1])
            ++k;
        float sum = 0, maximum = v[0];
        for (int i = 0; i < k; i++) {
            v[i] = expf((v[i] - maximum) / temp);
            sum += v[i];
        }
        if (topp < 1.f) {
            float cumulative = 0;
            int keep = 0;
            do {
                cumulative += v[keep++];
            } while (keep < k && cumulative < topp * sum);
            k = keep;
            sum = cumulative;
        }
        uint64_t state = rng[b];
        rng[b] = state + 0x9e3779b97f4a7c15ULL;
        float target = ((splitmix(state) >> 40) + .5f) * (1.f / 16777216) * sum, accum = 0;
        int chosen = id[k - 1];
        for (int i = 0; i < k; i++) {
            accum += v[i];
            if (accum >= target) {
                chosen = id[i];
                break;
            }
        }
        if (cb == 0 && chosen == 2051) {
            done[b] = 1;
            chosen = 0;
        }
        if (cb == 0 && !done[b] && seen)
            seen[b * 2048 + chosen] = 1;
        codes[b * 16 + cb] = chosen;
    }
}
void sample(cudaStream_t stream, TensorView logits, int *codes, uint64_t *rng, int batch,
            int branches, int cb, float cfg, float temp, int topk, int *done, float topp,
            float cfg_ref, float cfg_ins, float penalty, int *seen) {
    if (temp == 0) {
        if (logits.type == Type::BF16)
            greedy_kernel<<<batch, 256, 0, stream>>>(logits.ptr<bf16>(), logits.cols, codes, batch,
                                                     branches, cb, cfg, done, cfg_ref, cfg_ins,
                                                     penalty, seen);
        else
            greedy_kernel<<<batch, 256, 0, stream>>>(logits.ptr<float>(), logits.cols, codes, batch,
                                                     branches, cb, cfg, done, cfg_ref, cfg_ins,
                                                     penalty, seen);
    } else {
        if (logits.type == Type::BF16)
            sample_kernel<<<batch, 256, 0, stream>>>(logits.ptr<bf16>(), logits.cols, codes, rng,
                                                     batch, branches, cb, cfg, temp, topk, done,
                                                     topp, cfg_ref, cfg_ins, penalty, seen);
        else
            sample_kernel<<<batch, 256, 0, stream>>>(logits.ptr<float>(), logits.cols, codes, rng,
                                                     batch, branches, cb, cfg, temp, topk, done,
                                                     topp, cfg_ref, cfg_ins, penalty, seen);
    }
}
} // namespace tts
