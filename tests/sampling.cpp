#include "sampling.h"
#include "audio_kernels.h"
#include "backend.h"
#include <cassert>
#include <cuda_bf16.h>
#include <iostream>
// Synthetic distributions isolate CFG/penalty/filtering from model numerics.
int main() {
    tts::Backend g;
    int batch = 2, branches = 3, vocab = 2052;
    tts::Buffer logits(size_t(batch) * branches * vocab * 4, &g.memory), codes(batch * 16 * 4, &g.memory),
        done(batch * 4, &g.memory), rng(batch * 8, &g.memory), seen(batch * 2048 * 4, &g.memory);
    std::vector<float> l(size_t(batch) * branches * vocab, -100);
    for (int b = 0; b < batch; b++) {
        l[b * vocab + 2] = 4;
        l[b * vocab + 3] = 3;
        l[(batch + b) * vocab + 2] = 1;
        l[(batch + b) * vocab + 3] = 4;
        l[(2 * batch + b) * vocab + 2] = 5;
        l[(2 * batch + b) * vocab + 3] = 1;
    }
    std::vector<int> history(batch * 2048);
    history[2] = 1;
    std::vector<uint64_t> seeds{42, 43};
    g.upload(logits.data(), l.data(), logits.size());
    g.upload(seen.data(), history.data(), seen.size());
    g.upload(rng.data(), seeds.data(), rng.size());
    g.zero(done);
    g.zero(codes);
    // dual scales 0,1 select instruction branch; penalty makes lane0 select token3.
    l[2 * batch * vocab + 3] = 3;
    l[(2 * batch + 1) * vocab + 3] = 3;
    g.upload(logits.data(), l.data(), logits.size());
    tts::sample(g.stream, logits.view(batch * branches, vocab, tts::Type::F32), {16, 2048, 2051},
                codes.ptr<int>(), rng.ptr<uint64_t>(), batch, branches, 0, 1, 0, 50,
                done.ptr<int>(), 1, 0, 1, 2, seen.ptr<int>());
    std::vector<int> out(batch * 16);
    g.download(out.data(), codes.data(), codes.size());
    assert(out[0] == 3 && out[16] == 2);
    // top-p tiny includes only the largest code, regardless of RNG; depth uses
    // distinct branch controls and doesn't read/update backbone repetition history.
    tts::sample(g.stream, logits.view(batch * branches, vocab, tts::Type::F32), {16, 2048, 2051},
                codes.ptr<int>(), rng.ptr<uint64_t>(), batch, branches, 1, 1, .9, 0,
                done.ptr<int>(), .01, 2, 0, 2, seen.ptr<int>());
    g.download(out.data(), codes.data(), codes.size());
    assert(out[1] == 3 && out[17] == 3);
    // Top-k zero retains all valid tokens but excludes 2048..2050. Backbone EOS remains.
    l.assign(l.size(), -100);
    l[2051] = 100;
    l[vocab + 2048] = 1000;
    l[vocab + 7] = 90;
    g.upload(logits.data(), l.data(), logits.size());
    tts::sample(g.stream, logits.view(batch, vocab, tts::Type::F32), {16, 2048, 2051},
                codes.ptr<int>(), rng.ptr<uint64_t>(), batch, 1, 0, 1, 1, 1, done.ptr<int>());
    g.download(out.data(), codes.data(), codes.size());
    std::vector<int> ended(batch);
    g.download(ended.data(), done.data(), done.size());
    assert(ended[0] && out[0] == 0 && out[16] == 7 && !ended[1]);
    // Non-checkpoint vocabularies and frame strides exercise both sampling paths.
    for (auto layout : {tts::SamplingLayout{3, 37, 40}, tts::SamplingLayout{7, 127, 130}}) {
        int width = layout.eos + 1;
        std::vector<float> values(batch * width, -100);
        values[layout.eos] = 100;
        values[width + layout.entries] = 1000; // Invalid codec ID must stay masked.
        values[width + layout.entries - 1] = 90;
        for (float temperature : {0.f, 1.f}) {
            std::vector<int> sentinel(batch * layout.codebooks + 4, -7);
            g.upload(logits.data(), values.data(), values.size() * sizeof(float));
            g.upload(codes.data(), sentinel.data(), sentinel.size() * sizeof(int));
            g.zero(done);
            tts::sample(g.stream, logits.view(batch, width, tts::Type::F32), layout,
                        codes.ptr<int>(), rng.ptr<uint64_t>(), batch, 1, 0, 1, temperature, 1,
                        done.ptr<int>());
            g.download(sentinel.data(), codes.data(), sentinel.size() * sizeof(int));
            g.download(ended.data(), done.data(), done.size());
            assert(ended[0] && !ended[1] && sentinel[0] == 0);
            assert(sentinel[layout.codebooks] == layout.entries - 1);
            for (size_t i = 0; i < sentinel.size(); ++i)
                if (i != 0 && i != size_t(layout.codebooks))
                    assert(sentinel[i] == -7);
        }
    }
    // Embedding must honor arbitrary codebook layouts and a strided output view.
    constexpr int books = 3, entries = 41, hidden = 10, stride = 2 * hidden;
    std::vector<__nv_bfloat16> table(books * entries * hidden), result(batch * branches * stride);
    std::vector<int> selected{1, 4, 7, 2, 5, 8};
    for (size_t i = 0; i < table.size(); ++i)
        table[i] = __nv_bfloat16(float(i % 17));
    tts::Buffer embedding(table.size() * sizeof(__nv_bfloat16), &g.memory),
        output(result.size() * sizeof(__nv_bfloat16), &g.memory);
    g.upload(embedding.data(), table.data(), embedding.size());
    g.upload(codes.data(), selected.data(), selected.size() * sizeof(int));
    for (int cb : {-1, 1}) {
        std::fill(result.begin(), result.end(), __nv_bfloat16(-7.f));
        g.upload(output.data(), result.data(), output.size());
        tts::audio_embedding(g.stream, embedding.view(books * entries, hidden), codes.ptr<int>(),
                             output.view(batch * branches, stride).columns(hidden, hidden), batch,
                             branches, books, cb);
        g.download(result.data(), output.data(), output.size());
        for (int row = 0; row < batch * branches; ++row)
            for (int d = 0; d < hidden; ++d) {
                float sum = 0;
                for (int k = cb < 0 ? 0 : cb; k < (cb < 0 ? books : cb + 1); ++k)
                    sum += float(
                        table[(k * entries + selected[(row % batch) * books + k]) * hidden + d]);
                assert(float(result[row * stride + hidden + d]) == sum);
                assert(float(result[row * stride + d]) == -7.f);
            }
    }
    std::cout << "sampling CFG, penalty, top-p, masking, EOS and embedding layouts passed\n";
}
