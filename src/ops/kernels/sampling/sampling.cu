#include "runtime/cuda_context.h"
#include "ops/sampling.h"
#include "ops/kernels/common/cuda_math.cuh"
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_radix_sort.cuh>
namespace tts {
struct LogitMaximum {
    __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
};
// Match PyTorch's CUDA exponential_ TensorIterator mapping, including the
// unused float4 components and the per-call Philox offset increment.
__device__ float exponential_noise(SamplingRng random, int token, int random_threads) {
    curandStatePhilox4_32_10_t state;
    curand_init(random.seed, token % random_threads,
                random.offset + uint64_t(token / (random_threads * 4)) * 4, &state);
    float4 values = curand_uniform4(&state);
    int component = token / random_threads % 4;
    float uniform = component < 2 ? (component ? values.y : values.x)
                                  : (component == 2 ? values.z : values.w);
    return uniform >= 1.f - FLT_EPSILON / 2 ? FLT_EPSILON / 2 : -logf(uniform);
}
template <class T>
__device__ float multinomial_score(float probability, SamplingRng random, int token, int threads) {
    return rounded<T>(probability / rounded<T>(exponential_noise(random, token, threads)));
}
__device__ void advance_random(SamplingRng &random, int vocabulary, int threads) {
    random.offset += uint64_t((vocabulary - 1) / (threads * 4) + 1) * 4;
}
static int random_threads(const CudaContext &cuda, int vocabulary) {
    return int(std::min((int64_t(vocabulary) + 255) / 256, int64_t(cuda.resident_threads / 256))) * 256;
}
template <class T>
__global__ void multinomial_kernel(const T *probabilities, int rows_stride, int vocabulary,
                                    int64_t *indices, SamplingRng *random_states, int threads) {
    using Candidate = cub::KeyValuePair<int, float>;
    using Select = cub::BlockReduce<Candidate, 256>;
    __shared__ typename Select::TempStorage storage;
    int row = blockIdx.x;
    SamplingRng random = random_states[row];
    Candidate best{INT_MAX, -INFINITY};
    for (int token = threadIdx.x; token < vocabulary; token += blockDim.x)
        best = cub::ArgMax{}(best, Candidate{token,
            multinomial_score<T>(f(probabilities[size_t(row) * rows_stride + token]), random, token, threads)});
    best = Select(storage).Reduce(best, cub::ArgMax{});
    if (!threadIdx.x) {
        indices[row] = best.key;
        advance_random(random_states[row], vocabulary, threads);
    }
}
void ops::multinomial(TensorView probabilities, TensorView indices, TensorView random_states) {
    auto &cuda = execution_context(probabilities, {indices, random_states});
    if (probabilities.rows < 1 || probabilities.cols < 1 ||
        indices.type != Type::I64 || indices.rows != probabilities.rows || indices.cols != 1 || !indices.dense() ||
        random_states.type != Type::I64 || random_states.rows != probabilities.rows || random_states.cols != 2 || !random_states.dense() ||
        probabilities.overlaps(indices) || probabilities.overlaps(random_states) || indices.overlaps(random_states))
        throw std::invalid_argument("multinomial probability/output/RNG shape or storage mismatch");
    auto launch = [&](auto value) {
        using T = decltype(value);
        multinomial_kernel<<<probabilities.rows, 256, 0, cuda.stream>>>(probabilities.ptr<T>(),
            probabilities.ld(), probabilities.cols, indices.ptr<int64_t>(), random_states.ptr<SamplingRng>(),
            random_threads(cuda, probabilities.cols));
    };
    if (probabilities.type == Type::BF16) launch(bf16{});
    else if (probabilities.type == Type::F16) launch(__half{});
    else if (probabilities.type == Type::F32) launch(float{});
    else throw std::invalid_argument("multinomial requires F32, F16 or BF16 probabilities");
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
__global__ void sample_kernel(const T *logits, int vocab, int *codes, SamplingRng *rng, int batch,
                              int branches, int cb, float cfg, float temp, int topk, int *done,
                              float topp, float cfg_ref, float cfg_ins, float penalty, int *seen,
                              SamplingLayout layout, const SamplingRow *rows, int random_threads) {
    using Sort = cub::BlockRadixSort<float, 256, 16, int>;
    using Reduce = cub::BlockReduce<float, 256>;
    using Candidate = cub::KeyValuePair<int, float>;
    using Select = cub::BlockReduce<Candidate, 256>;
    __shared__ union {
        typename Sort::TempStorage sort;
        typename Reduce::TempStorage reduce;
        typename Select::TempStorage select;
        struct {
            float values[4096];
            int ids[4096];
        } ordered;
    } storage;
    __shared__ int candidates;
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
        else if (temp > 0) values[j] /= temp;
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
        int valid = layout.entries + (cb == 0 && !suppress_eos ? 1 : 0),
            limit = valid, k = topk > 0 ? min(topk, valid) : valid;
        if (!nucleus_before_temperature && cb == 0)
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
                ? rounded<T>(v[i] / temp) - scaled_maximum : v[i] - maximum);
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
        candidates = k;
    }
    __syncthreads();
    const SamplingRng random = rng[b];
    Candidate best{INT_MAX, -INFINITY};
    for (int index = threadIdx.x; index < candidates; index += blockDim.x) {
        int token = storage.ordered.ids[index];
        float probability = storage.ordered.values[index];
        float score = nucleus_before_temperature
            ? multinomial_score<T>(probability, random, token, random_threads)
            : multinomial_score<float>(probability, random, token, random_threads);
        best = cub::ArgMax{}(best, Candidate{token, score});
    }
    // Every thread has consumed the ordered values before reduction reuses them.
    __syncthreads();
    best = Select(storage.select).Reduce(best, cub::ArgMax{});
    if (!threadIdx.x) {
        advance_random(rng[b], vocab, random_threads);
        int chosen = best.key;
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
void sample( TensorView logits, SamplingLayout layout, int *codes,
            SamplingRng *rng, int batch, int branches, int cb, float cfg, float temp, int topk,
            int *done, float topp, float cfg_ref, float cfg_ins, float penalty, int *seen,
            const SamplingRow *rows) {
    const auto &cuda = execution_context(logits);
    const auto stream = cuda.stream;
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
                branches, cb, cfg, temp, topk, done, topp, cfg_ref, cfg_ins, penalty, seen, layout, rows,
                random_threads(cuda, logits.cols));
    };
    if (logits.type == Type::BF16) launch(bf16{});
    else if (logits.type == Type::F16) launch(__half{});
    else launch(float{});
}
} // namespace tts
