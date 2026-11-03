#include "ops/sampling.h"
#include "ops/kernels/common/cuda_math.cuh"
#include <cuda_fp16.h>
#include <cmath>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_radix_sort.cuh>
namespace tts {
struct LogitMaximum {
    __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
};
__device__ uint64_t splitmix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
template <class T>
__device__ float guided_logit(const T *logits, int vocab, int b, int batch, int branches, int cb,
                              int id, float cfg, float cfg_ref, float cfg_ins, float penalty,
                              const int *seen, SamplingLayout layout,
                              const int *logit_rows = nullptr, bool suppress_eos = false,
                              float *raw = nullptr) {
    if (raw) *raw = -INFINITY;
    if (id >= vocab || !(id < layout.entries || (cb == 0 && id == layout.eos && !suppress_eos)))
        return -INFINITY;
    float v = f(logits[(logit_rows ? logit_rows[0] : b) * vocab + id]);
    if (branches == 2) {
        float u = f(logits[(logit_rows ? logit_rows[1] : batch + b) * vocab + id]);
        v = u + cfg * (v - u);
    }
    if (branches == 3) {
        float r = f(logits[(logit_rows ? logit_rows[1] : batch + b) * vocab + id]),
              ins = f(logits[(logit_rows ? logit_rows[2] : 2 * batch + b) * vocab + id]);
        v = v + cfg_ref * (r - v) + cfg_ins * (ins - v);
    }
    if (raw) *raw = v;
    if (cb == 0 && id < layout.entries && seen && seen[b * layout.entries + id])
        v = v < 0 ? v * penalty : v / penalty;
    return v;
}
template <class T>
__device__ void greedy(const T *logits, int vocab, int *codes, int batch, int branches, int cb,
                       float cfg, int *done, float cfg_ref, float cfg_ins, float penalty, int *seen,
                       SamplingLayout layout, int b, const int *logit_rows, bool suppress_eos = false) {
    __shared__ float vals[8];
    __shared__ int ids[8];
    int id = 0;
    float val = -INFINITY;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        float v = guided_logit(logits, vocab, b, batch, branches, cb, i, cfg, cfg_ref, cfg_ins,
                               penalty, seen, layout, logit_rows, suppress_eos);
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
        if (cb == 0 && id == layout.eos) {
            done[b] = 1;
            id = 0;
        }
        if (cb == 0 && !done[b] && seen)
            seen[b * layout.entries + id] = 1;
        codes[b * layout.codebooks + cb] = id;
    }
}
template <class T>
__global__ void greedy_kernel(const T *logits, int vocab, int *codes, int batch, int branches,
                              int cb, float cfg, int *done, float cfg_ref, float cfg_ins,
                              float penalty, int *seen, SamplingLayout layout) {
    greedy(logits, vocab, codes, batch, branches, cb, cfg, done, cfg_ref, cfg_ins, penalty, seen,
           layout, int(blockIdx.x), nullptr);
}
template <class T>
__global__ void sample_kernel(const T *logits, int vocab, int *codes, uint64_t *rng, int batch,
                              int branches, int cb, float cfg, float temp, int topk, int *done,
                              float topp, float cfg_ref, float cfg_ins, float penalty, int *seen,
                              SamplingLayout layout, const SamplingRow *rows) {
    using Sort = cub::BlockRadixSort<float, 256, 16, int>;
    using Reduce = cub::BlockReduce<float, 256>;
    __shared__ union {
        typename Sort::TempStorage sort;
        typename Reduce::TempStorage reduce;
        struct {
            float values[4096];
            int ids[4096];
        } ordered;
    } storage;
    int b = blockIdx.x;
    int *batch_done = rows ? done : nullptr;
    const int *logit_rows = nullptr;
    bool suppress_eos = false, eos_on_argmax = false, nucleus_before_temperature = false,
         include_threshold_token = true;
    if (rows) {
        const auto &row = rows[b];
        if (!row.codes) {
            if (!threadIdx.x && cb == 0 && batch_done)
                batch_done[b] = 1;
            return;
        }
        const auto &params = cb == 0 ? row.backbone : row.depth;
        codes = row.codes;
        done = row.done;
        seen = cb == 0 ? row.seen : nullptr;
        rng = row.rng;
        branches = row.branches;
        cfg = row.cfg;
        cfg_ref = row.cfg_ref;
        cfg_ins = row.cfg_ins;
        penalty = row.penalty;
        temp = params.temperature;
        topp = params.top_p;
        topk = params.top_k;
        nucleus_before_temperature = params.nucleus_before_temperature;
        include_threshold_token = params.include_threshold_token;
        logit_rows = row.logits;
        suppress_eos = cb == 0 && row.generated && *row.generated < row.eos_min_tokens;
        eos_on_argmax = cb == 0 && row.eos_on_argmax && !suppress_eos;
        b = 0;
    }
    if (temp == 0 && !eos_on_argmax) {
        greedy(logits, vocab, codes, batch, branches, cb, cfg, done, cfg_ref, cfg_ins, penalty,
               seen, layout, b, logit_rows, suppress_eos);
        if (!threadIdx.x && cb == 0 && batch_done)
            batch_done[blockIdx.x] = *done;
        return;
    }
    float values[16];
    int ids[16];
    float raw_max = -INFINITY;
#pragma unroll
    for (int j = 0; j < 16; j++) {
        int id = threadIdx.x * 16 + j;
        ids[j] = id;
        float raw;
        values[j] = guided_logit(logits, vocab, b, batch, branches, cb, id, cfg, cfg_ref, cfg_ins,
                                 penalty, seen, layout, logit_rows, suppress_eos, &raw);
        if (nucleus_before_temperature) values[j] = rounded<T>(values[j]);
        if (id != layout.eos) raw_max = fmaxf(raw_max, raw);
    }
    if (eos_on_argmax) {
        raw_max = Reduce(storage.reduce).Reduce(raw_max, LogitMaximum{});
        __shared__ bool stop;
        if (!threadIdx.x) {
            // EOS follows all normal entries, so ties select the normal token,
            // matching the first-index rule of argmax.
            stop = guided_logit(logits, vocab, b, batch, branches, cb, layout.eos, cfg,
                cfg_ref, cfg_ins, 1, nullptr, layout, logit_rows) > raw_max;
            if (stop) {
                done[b] = 1;
                codes[b * layout.codebooks + cb] = 0;
                if (batch_done) batch_done[blockIdx.x] = 1;
            }
        }
        __syncthreads();
        if (stop) return;
    }
    if (temp == 0) {
        greedy(logits, vocab, codes, batch, branches, cb, cfg, done, cfg_ref, cfg_ins, penalty,
               seen, layout, b, logit_rows, suppress_eos);
        if (!threadIdx.x && cb == 0 && batch_done) batch_done[blockIdx.x] = *done;
        return;
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
        int valid = layout.entries + (cb == 0 && !suppress_eos ? 1 : 0),
            limit = valid, k = topk > 0 ? min(topk, valid) : valid;
        if (!nucleus_before_temperature)
            while (k < valid && v[k] == v[k - 1]) ++k;
        float sum = 0, maximum = v[0];
        if (nucleus_before_temperature && topp < 1.f) {
            // Some models apply nucleus filtering to the original distribution,
            // before temperature/top-k, and exclude the threshold-crossing token.
            for (int i = 0; i < valid; ++i) sum += expf(v[i] - maximum);
            float cumulative = 0;
            int keep = 0;
            for (int i = 0; i < valid; ++i) {
                cumulative += rounded<T>(expf(v[i] - maximum) / sum);
                if (rounded<T>(cumulative) > topp) {
                    if (include_threshold_token) ++keep;
                    break;
                }
                ++keep;
            }
            limit = max(1, keep);
        }
        if (nucleus_before_temperature) {
            temp = fmaxf(temp, 1.e-5f);
            k = topk > 0 ? min(topk, limit) : limit;
            float pivot = rounded<T>(v[k - 1] / temp);
            while (k < limit && rounded<T>(v[k] / temp) == pivot) ++k;
        }
        sum = 0;
        float scaled_maximum = rounded<T>(maximum / temp);
        for (int i = 0; i < k; i++) {
            v[i] = expf(nucleus_before_temperature
                ? rounded<T>(v[i] / temp) - scaled_maximum : (v[i] - maximum) / temp);
            sum += v[i];
        }
        if (!nucleus_before_temperature && topp < 1.f) {
            float cumulative = 0;
            int keep = 0;
            do {
                cumulative += v[keep++];
            } while (keep < k && cumulative < topp * sum);
            k = keep;
            sum = cumulative;
        }
        if (nucleus_before_temperature) {
            float total = 0;
            for (int i = 0; i < k; ++i) {
                v[i] = rounded<T>(v[i] / sum);
                total += v[i];
            }
            sum = total;
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
        if (cb == 0 && chosen == layout.eos) {
            done[b] = 1;
            chosen = 0;
        }
        if (cb == 0 && !done[b] && seen)
            seen[b * layout.entries + chosen] = 1;
        codes[b * layout.codebooks + cb] = chosen;
        if (cb == 0 && batch_done)
            batch_done[blockIdx.x] = *done;
    }
}
void sample(cudaStream_t stream, TensorView logits, SamplingLayout layout, int *codes,
            uint64_t *rng, int batch, int branches, int cb, float cfg, float temp, int topk,
            int *done, float topp, float cfg_ref, float cfg_ins, float penalty, int *seen,
            const SamplingRow *rows) {
    if (layout.codebooks < 1 || cb < 0 || cb >= layout.codebooks || layout.entries < 1 ||
        layout.entries > logits.cols ||
        (cb == 0 && (layout.eos < layout.entries || layout.eos >= logits.cols)) ||
        ((temp != 0 || rows) && logits.cols > 4096))
        throw std::invalid_argument("sampling layout exceeds logits or the block-sort capacity");
    if (logits.type != Type::BF16 && logits.type != Type::F16 && logits.type != Type::F32)
        throw std::invalid_argument("invalid sampling dtype");
    auto launch = [&](auto value) {
        using T = decltype(value);
        if (temp == 0 && !rows)
            greedy_kernel<<<batch, 256, 0, stream>>>(logits.ptr<T>(), logits.cols, codes, batch,
                branches, cb, cfg, done, cfg_ref, cfg_ins, penalty, seen, layout);
        else
            sample_kernel<<<batch, 256, 0, stream>>>(logits.ptr<T>(), logits.cols, codes, rng, batch,
                branches, cb, cfg, temp, topk, done, topp, cfg_ref, cfg_ins, penalty, seen, layout, rows);
    };
    if (logits.type == Type::BF16) launch(bf16{});
    else if (logits.type == Type::F16) launch(__half{});
    else launch(float{});
}
} // namespace tts
