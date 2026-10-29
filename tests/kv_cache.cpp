#include "kv_cache.h"
#include "backend.h"
#include <cuda_bf16.h>
#include <cassert>
#include <cmath>
#include <iostream>

using namespace tts;
static void check_paged(Backend &g, int page_size) {
    constexpr int batch = 2, cap = 512, nh = 4, nk = 2, dim = 128;
    const int blocks = cap / page_size, pages = batch * blocks;
    const int lengths[] = {129, 259};
    int rows = lengths[0] + lengths[1];
    Buffer input(size_t(rows) * (nh + 2 * nk) * dim * 2, &g.memory), q(size_t(rows) * nh * dim * 2, &g.memory),
        k(size_t(batch) * cap * nk * dim * 2, &g.memory), v(k.size(), &g.memory), pk(k.size(), &g.memory), pv(k.size(), &g.memory), a(q.size(), &g.memory),
        b(q.size(), &g.memory), table(pages * sizeof(int), &g.memory), meta((2 * rows + 7) * 4, &g.memory), pos(8, &g.memory),
        rope(size_t(cap) * dim * 2, &g.memory);
    // Deliberately nonconsecutive, nonmonotone physical pages; lanes own disjoint pages.
    std::vector<int> mapping(pages);
    for (int i = 0; i < pages; ++i)
        mapping[i] = (5 * i + 3) % pages;
    std::vector<int> metadata(2 * rows + 7);
    int row = 0;
    for (int lane = 0; lane < batch; ++lane) {
        metadata[2 * rows + lane] = row;
        metadata[2 * rows + 3 + lane] = lengths[lane];
        for (int p = 0; p < lengths[lane]; ++p, ++row) {
            metadata[row] = lane;
            metadata[rows + row] = p;
        }
    }
    metadata[2 * rows + 2] = rows;
    std::vector<__nv_bfloat16> data(input.size() / 2);
    for (size_t i = 0; i < data.size(); ++i)
        data[i] = __float2bfloat16(std::sin(i * .013) * .5);
    g.upload(input.data(), data.data(), input.size());
    g.upload(table.data(), mapping.data(), mapping.size() * sizeof(int));
    g.reserve_attention_workspace(batch * cap * nh);
    g.upload(meta.data(), metadata.data(), meta.size());
    g.make_rope(rope.view(cap, dim), dim, 1000000, 1, 0);
    PackedBatch packed{meta.ptr<int>(), meta.ptr<int>() + rows, meta.ptr<int>() + 2 * rows,
                       meta.ptr<int>() + 2 * rows + 3, rows};
    CacheLayout paged{table.ptr<int>(), blocks, page_size, false};
    auto write = [&](Buffer &key, Buffer &val, CacheLayout layout, int n, int time,
                     const int *positions, const PackedBatch *pack) {
        g.qkv(input.view(n, (nh + 2 * nk) * dim), q.view(n, nh * dim),
              key.view(batch * cap, nk * dim), val.view(batch * cap, nk * dim), positions, batch,
              time, 0, cap, nh, nk, dim, {}, {}, 1e-6f, false, rope.view(cap, dim), 0, pack,
              layout);
    };
    write(k, v, {}, rows, lengths[1], nullptr, &packed);
    write(pk, pv, paged, rows, lengths[1], nullptr, &packed);
    auto attend = [&](Buffer &key, Buffer &val, Buffer &out, CacheLayout layout, int n, int time,
                      const int *positions, const PackedBatch *pack) {
        g.attention(q.view(n, nh * dim), key.view(batch * cap, nk * dim),
                    val.view(batch * cap, nk * dim), out.view(n, nh * dim), positions, batch, time,
                    0, cap, nh, nk, dim, true, 0, 0, pack, layout);
    };
    attend(k, v, a, {}, rows, lengths[1], nullptr, &packed);
    attend(pk, pv, b, paged, rows, lengths[1], nullptr, &packed);
    std::vector<__nv_bfloat16> x(rows * nh * dim), y(x.size());
    g.download(x.data(), a.data(), x.size() * 2);
    g.download(y.data(), b.data(), y.size() * 2);
    float error = 0;
    for (size_t i = 0; i < x.size(); ++i)
        error = std::max(error, std::abs(float(x[i]) - float(y[i])));
    assert(error <= .004f);
    // Short, non-packed prefill exercises the shared page-table fallback too.
    attend(k, v, a, {}, batch * 31, 31, nullptr, nullptr);
    attend(pk, pv, b, paged, batch * 31, 31, nullptr, nullptr);
    g.download(x.data(), a.data(), batch * 31 * nh * dim * 2);
    g.download(y.data(), b.data(), batch * 31 * nh * dim * 2);
    for (int i = 0; i < batch * 31 * nh * dim; ++i)
        assert(x[i] == y[i]);
    // Appends straddle another page boundary, with independent absolute lane lengths.
    float decode_error = 0;
    for (int step = 0; step < 132; ++step) {
        int positions[] = {lengths[0] + step, lengths[1] + step};
        g.upload(pos.data(), positions, sizeof(positions));
        write(k, v, {}, batch, 1, pos.ptr<int>(), nullptr);
        write(pk, pv, paged, batch, 1, pos.ptr<int>(), nullptr);
        attend(k, v, a, {}, batch, 1, pos.ptr<int>(), nullptr);
        attend(pk, pv, b, paged, batch, 1, pos.ptr<int>(), nullptr);
        g.download(x.data(), a.data(), batch * nh * dim * 2);
        g.download(y.data(), b.data(), batch * nh * dim * 2);
        for (int i = 0; i < batch * nh * dim; ++i)
            decode_error = std::max(decode_error, std::abs(float(x[i]) - float(y[i])));
    }
    // FA rounds softmax probabilities to BF16 for tensor-core PV; the dense
    // oracle accumulates FP32. Inputs/values have magnitude <= 0.5.
    assert(decode_error <= .004f);
    std::cout << "page size " << page_size << ": paged prefill max error " << error
              << "; paged decode max error " << decode_error << "\n";
}
static void check_ring(Backend &g) {
    constexpr int batch = 2, nh = 4, nk = 4, dim = 64, cap = 256, window = 72;
    Buffer input(batch * (nh + 2 * nk) * dim * 4, &g.memory), q(batch * nh * dim * 4, &g.memory),
        k(batch * cap * nk * dim * 4, &g.memory), v(k.size(), &g.memory), rk(batch * window * nk * dim * 4, &g.memory),
        rv(rk.size(), &g.memory), a(q.size(), &g.memory), b(q.size(), &g.memory), pos(batch * 4, &g.memory), rope(cap * dim * 4, &g.memory);
    std::vector<float> data(input.size() / 4), x(a.size() / 4), y(x.size());
    g.make_rope(rope.view(cap, dim, Type::F32), dim, 10000, 1, 0);
    for (int step = 0; step < 151; ++step) {
        for (size_t i = 0; i < data.size(); ++i)
            data[i] = std::sin(i * .017 + step * .09);
        int positions[] = {step, step};
        g.upload(input.data(), data.data(), input.size());
        g.upload(pos.data(), positions, sizeof(positions));
        auto run = [&](Buffer &key, Buffer &value, Buffer &out, int capacity, bool ring) {
            CacheLayout layout;
            layout.ring = ring;
            g.qkv(input.view(batch, (nh + 2 * nk) * dim, Type::F32),
                  q.view(batch, nh * dim, Type::F32),
                  key.view(batch * capacity, nk * dim, Type::F32),
                  value.view(batch * capacity, nk * dim, Type::F32), pos.ptr<int>(), batch, 1, 0,
                  capacity, nh, nk, dim, {}, {}, 1e-6f, false, rope.view(cap, dim, Type::F32), -1,
                  nullptr, layout);
            g.attention(q.view(batch, nh * dim, Type::F32),
                        key.view(batch * capacity, nk * dim, Type::F32),
                        value.view(batch * capacity, nk * dim, Type::F32),
                        out.view(batch, nh * dim, Type::F32), pos.ptr<int>(), batch, 1, 0, capacity,
                        nh, nk, dim, true, window, -1, nullptr, layout);
        };
        run(k, v, a, cap, false);
        run(rk, rv, b, window, true);
        g.download(x.data(), a.data(), a.size());
        g.download(y.data(), b.data(), b.size());
        assert(x == y);
    }
    std::cout << "72-slot ring matches dense KV bitwise through two wraps\n";
}
static void check_short_attention(Backend &g) {
    constexpr int batch = 2, nh = 8, nk = 2, dim = 128;
    Buffer q(batch * 2 * nh * dim * 2, &g.memory), k(batch * 31 * nk * dim * 2, &g.memory), v(k.size(), &g.memory), a(q.size(), &g.memory),
        b(q.size(), &g.memory), positions(batch * sizeof(int), &g.memory);
    std::vector<__nv_bfloat16> data(k.size() / 2), x(q.size() / 2), y(x.size());
    for (size_t i = 0; i < data.size(); ++i)
        data[i] = __float2bfloat16(std::sin(i * .037) * .5);
    g.upload(k.data(), data.data(), k.size());
    g.upload(v.data(), data.data(), v.size());
    g.upload(q.data(), data.data(), q.size());
    for (int cap : {16, 17, 31})
        for (int end : {15, 16, 17}) {
            if (end > cap)
                continue;
            int p[] = {end - 2, end - 2};
            g.upload(positions.data(), p, sizeof(p));
            auto run = [&](Buffer &out, const int *pos) {
                g.attention(q.view(batch * 2, nh * dim), k.view(batch * cap, nk * dim),
                            v.view(batch * cap, nk * dim), out.view(batch * 2, nh * dim), pos,
                            batch, 2, 0, cap, nh, nk, dim, true, 0, end - 2);
            };
            run(a, nullptr);
            run(b, positions.ptr<int>()); // independent scalar softmax path
            g.download(x.data(), a.data(), a.size());
            g.download(y.data(), b.data(), b.size());
            for (size_t i = 0; i < x.size(); ++i)
                assert(std::abs(float(x[i]) - float(y[i])) <= .004f);
        }
    std::cout << "short attention agrees across capacities 16/17/31 and key counts 15/16/17\n";
}
static void check_reuse(Backend &g) {
    PagedKVCache pool(g.memory, 2, 128, Type::BF16, 6, 512);
    pool.reserve(7);
    assert(pool.capacity() == 7);
    auto *address = pool.keys(0).data;
    for (const auto &lengths :
         std::vector<std::vector<int>>{{129, 259}, {1, 127, 128, 129}, {512}}) {
        PagedKVCache::Lease lease(pool, g.stream, lengths);
        assert(pool.keys(0).data == address);
    }
    bool rejected = false;
    try {
        PagedKVCache::Lease bad(pool, g.stream, {513});
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);
    pool.reserve(16);
    PagedKVCache::Lease lease(pool, g.stream, {512, 512, 512});
    PagedKVCache unaligned(g.memory, 1, 1, Type::BF16, 1, 130, 64);
    unaligned.reserve(3);
    assert(unaligned.required_pages({130}) == 3);
    for (const auto &lengths : std::vector<std::vector<int>>{{131}, {0}, {-1}, {}, {1, 1}}) {
        rejected = false;
        try {
            PagedKVCache::Lease bad(unaligned, g.stream, lengths);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        assert(rejected);
    }
    PagedKVCache::Lease valid(unaligned, g.stream, {130});
    // Pool geometry must not impose the engine's independent context admission limit.
    PagedKVCache larger(g.memory, 1, 1, Type::BF16, 1, 9001);
    assert(larger.required_pages({9001}) == 71);
    larger.reserve(71);
    PagedKVCache::Lease big(larger, g.stream, {9001});
    std::cout << "page leases, reuse, rejection recovery and idle growth passed\n";
}
int main() {
    Backend g;
    for (int page_size : {64, 128, 256})
        check_paged(g, page_size);
    check_ring(g);
    check_short_attention(g);
    check_reuse(g);
    // Allocation failure must be recoverable without poisoning the CUDA error state.
    DeviceArena arena(g.memory, 1 << 20);
    size_t available, total;
    CUDA(cudaMemGetInfo(&available, &total));
    bool rejected = false;
    try {
        arena.reserve(total + (size_t(1) << 30));
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    assert(rejected);
    arena.reserve(1 << 20);
    assert(arena.allocate(256));
    CUDA(cudaGetLastError());
    std::cout << "CUDA pool OOM recovery passed\n";
}
