#include "kernels/sampling/sampling.h"
#include "kernels/audio/audio_kernels.h"
#include "backends/cuda/backend.h"
#include <cassert>
#include <cuda_bf16.h>
#include <iostream>
static void mapped_sampling(tts::Backend &g) {
    constexpr int count = 3, books = 3, entries = 37, vocab = 41, lanes = 6;
    std::vector<float> logits(lanes * vocab);
    for (size_t i = 0; i < logits.size(); ++i)
        logits[i] = float(int((i * 17) % 29) - 14) * .17f;
    tts::Buffer input(logits.size() * sizeof(float), &g.memory),
        output(count * books * sizeof(int), &g.memory), done(count * sizeof(int), &g.memory),
        history(count * entries * sizeof(int), &g.memory), rng(count * sizeof(uint64_t), &g.memory),
        metadata((count + 1) * sizeof(tts::SamplingRow), &g.memory);
    std::vector<uint64_t> seeds{11, 23, 37};
    g.upload(input.data(), logits.data(), input.size());
    auto reset = [&] {
        g.zero(output);
        g.zero(done);
        g.zero(history);
        g.upload(rng.data(), seeds.data(), rng.size());
    };
    reset();
    std::vector<tts::SamplingRow> rows(count + 1); // Last row is inactive padding.
    for (int request = 0, offset = 0; request < count; ++request) {
        auto &row = rows[count - 1 - request]; // Dense scheduling order differs from state order.
        row.codes = output.ptr<int>() + request * books;
        row.done = done.ptr<int>() + request;
        row.seen = history.ptr<int>() + request * entries;
        row.rng = rng.ptr<uint64_t>() + request;
        row.branches = request + 1;
        row.cfg = 2;
        row.cfg_ref = .7f;
        row.cfg_ins = 1.3f;
        row.penalty = 1.1f;
        row.backbone = {request == 0 ? 0.f : .8f, .9f, 9};
        row.depth = {.6f, .85f, 7};
        for (int branch = 0; branch < row.branches; ++branch)
            row.logits[branch] = offset + branch;
        for (int cb = 0; cb < books; ++cb) {
            auto params = cb ? row.depth : row.backbone;
            tts::sample(g.stream,
                        input.view(lanes, vocab, tts::Type::F32).slice(offset, row.branches),
                        {books, entries, 40}, row.codes, row.rng, 1, row.branches, cb, row.cfg,
                        params.temperature, params.top_k, row.done, params.top_p, row.cfg_ref,
                        row.cfg_ins, row.penalty, cb ? nullptr : row.seen);
        }
        offset += row.branches;
    }
    std::vector<int> expected(count * books), actual(expected.size());
    std::vector<uint64_t> expected_rng(count), actual_rng(count);
    g.download(expected.data(), output.data(), output.size());
    g.download(expected_rng.data(), rng.data(), rng.size());
    reset();
    g.upload(metadata.data(), rows.data(), metadata.size());
    tts::Buffer dense_done((count + 1) * sizeof(int), &g.memory);
    std::vector<int> flags(count + 1, -1), state_flags(count);
    g.upload(dense_done.data(), flags.data(), dense_done.size());
    for (int cb = 0; cb < books; ++cb)
        tts::sample(g.stream, input.view(lanes, vocab, tts::Type::F32), {books, entries, 40},
                    nullptr, nullptr, count + 1, 1, cb, 1, 0, 0, dense_done.ptr<int>(), 1, 1, 1, 1,
                    nullptr, metadata.ptr<tts::SamplingRow>());
    g.download(actual.data(), output.data(), output.size());
    g.download(actual_rng.data(), rng.data(), rng.size());
    assert(actual == expected && actual_rng == expected_rng);
    g.download(flags.data(), dense_done.data(), dense_done.size());
    g.download(state_flags.data(), done.data(), done.size());
    assert(flags[count] == 1);
    for (int row = 0; row < count; ++row)
        assert(flags[row] == state_flags[count - 1 - row]);
}

// Synthetic distributions isolate CFG/penalty/filtering from model numerics.
int main() {
    tts::Backend g;
    mapped_sampling(g);
    int batch = 2, branches = 3, vocab = 2052;
    tts::Buffer logits(size_t(batch) * branches * vocab * 4, &g.memory),
        codes(batch * 16 * 4, &g.memory), done(batch * 4, &g.memory), rng(batch * 8, &g.memory),
        seen(batch * 2048 * 4, &g.memory);
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
