#include "backends/cuda/backend.h"
#include "kernels/attention/flash_attention.h"
#include <cassert>
#include <cmath>
#include <cuda_bf16.h>
#include <iostream>

using bf16 = __nv_bfloat16;

// Nonzero cached prefixes catch incorrect q/k lengths and causal alignment.
static void check(tts::Backend &g, int batch, int time, int length) {
    constexpr int heads = 8, kvheads = 2, dim = 128, capacity = 17;
    const int rows = batch * time;
    std::vector<bf16> q(size_t(rows) * heads * dim), k(size_t(batch) * capacity * kvheads * dim),
        v(k.size()), actual(q.size());
    for (size_t i = 0; i < q.size(); ++i)
        q[i] = bf16(.7f * std::sin(i * .037f));
    for (size_t i = 0; i < k.size(); ++i) {
        k[i] = bf16(.8f * std::cos(i * .019f));
        v[i] = bf16(.6f * std::sin(i * .071f));
    }
    tts::Buffer dq(q.size() * 2, &g.memory), dk(k.size() * 2, &g.memory),
        dv(v.size() * 2, &g.memory), out(dq.size(), &g.memory),
        lse(size_t(rows) * heads * sizeof(float), &g.memory);
    g.upload(dq.data(), q.data(), dq.size());
    g.upload(dk.data(), k.data(), dk.size());
    g.upload(dv.data(), v.data(), dv.size());
    std::vector<double> expected(q.size());
    for (int b = 0; b < batch; ++b)
        for (int t = 0; t < time; ++t)
            for (int h = 0; h < heads; ++h) {
                const int count = length - time + t + 1, kh = h / (heads / kvheads);
                const size_t query = (size_t(b * time + t) * heads + h) * dim;
                std::vector<double> scores(count);
                double maximum = -INFINITY, denominator = 0;
                for (int s = 0; s < count; ++s) {
                    const size_t key = ((size_t(b) * capacity + s) * kvheads + kh) * dim;
                    double dot = 0;
                    for (int d = 0; d < dim; ++d)
                        dot += float(q[query + d]) * double(float(k[key + d]));
                    scores[s] = dot / std::sqrt(double(dim));
                    maximum = std::max(maximum, scores[s]);
                }
                for (auto &score : scores) {
                    score = std::exp(score - maximum);
                    denominator += score;
                }
                for (int d = 0; d < dim; ++d)
                    for (int s = 0; s < count; ++s)
                        expected[query + d] +=
                            scores[s] / denominator *
                            float(v[((size_t(b) * capacity + s) * kvheads + kh) * dim + d]);
            }
    for (bool flash : {false, true}) {
        if (flash)
            tts::flash_prefill(g.stream, dq.view(rows, heads * dim),
                               dk.view(batch * capacity, kvheads * dim),
                               dv.view(batch * capacity, kvheads * dim),
                               out.view(rows, heads * dim), lse.ptr<float>(), batch, time, 0,
                               capacity, heads, kvheads, dim, true, 0, nullptr, {}, length);
        else
            g.attention(dq.view(rows, heads * dim), dk.view(batch * capacity, kvheads * dim),
                        dv.view(batch * capacity, kvheads * dim), out.view(rows, heads * dim),
                        nullptr, batch, time, 0, capacity, heads, kvheads, dim, true, 0,
                        length - time);
        g.download(actual.data(), out.data(), out.size());
        double error = 0, norm = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            assert(std::isfinite(float(actual[i])));
            const double delta = float(actual[i]) - expected[i];
            assert(std::abs(delta) < .008);
            error += delta * delta;
            norm += expected[i] * expected[i];
        }
        assert(error < .0001 * std::max(norm, 1e-30));
    }
}

int main() {
    tts::Backend g;
    for (int batch : {1, 3, 8})
        for (int length : {1, 2, 7, 16})
            for (int time : {1, 2})
                if (time <= length)
                    check(g, batch, time, length);
    std::cout << "short attention and FlashAttention match independent causal prefix oracle\n";
}
