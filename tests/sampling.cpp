#include "sampling.h"
#include "backend.h"
#include <cassert>
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
    tts::sample(g.stream, logits.view(batch * branches, vocab, tts::Type::F32), codes.ptr<int>(),
             rng.ptr<uint64_t>(), batch, branches, 0, 1, 0, 50, done.ptr<int>(), 1, 0, 1, 2,
             seen.ptr<int>());
    std::vector<int> out(batch * 16);
    g.download(out.data(), codes.data(), codes.size());
    assert(out[0] == 3 && out[16] == 2);
    // top-p tiny includes only the largest code, regardless of RNG; depth uses
    // distinct branch controls and doesn't read/update backbone repetition history.
    tts::sample(g.stream, logits.view(batch * branches, vocab, tts::Type::F32), codes.ptr<int>(),
             rng.ptr<uint64_t>(), batch, branches, 1, 1, .9, 0, done.ptr<int>(), .01, 2, 0, 2,
             seen.ptr<int>());
    g.download(out.data(), codes.data(), codes.size());
    assert(out[1] == 3 && out[17] == 3);
    // Top-k zero retains all valid tokens but excludes 2048..2050. Backbone EOS remains.
    l.assign(l.size(), -100);
    l[2051] = 100;
    l[vocab + 2048] = 1000;
    l[vocab + 7] = 90;
    g.upload(logits.data(), l.data(), logits.size());
    tts::sample(g.stream, logits.view(batch, vocab, tts::Type::F32), codes.ptr<int>(), rng.ptr<uint64_t>(),
             batch, 1, 0, 1, 1, 1, done.ptr<int>());
    g.download(out.data(), codes.data(), codes.size());
    std::vector<int> ended(batch);
    g.download(ended.data(), done.data(), done.size());
    assert(ended[0] && out[0] == 0 && out[16] == 7 && !ended[1]);
    std::cout << "sampling CFG, penalty, top-p, masking and EOS passed\n";
}
