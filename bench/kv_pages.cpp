#include "backend.h"
#include "graph.h"
#include "kv_cache.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>

// Replays all 28 layers, so long-context K/V exceed L2. No model load or host
// allocation is included in the CUDA-event measurement. Initialization is zero.
int main(int argc, char **argv) {
    try {
        int page = argc > 1 ? std::atoi(argv[1]) : 128;
        int batch = argc > 2 ? std::atoi(argv[2]) : 8;
        int context = argc > 3 ? std::atoi(argv[3]) : 2049;
        int repeats = argc > 4 ? std::atoi(argv[4]) : 30;
        if (batch < 1 || batch > 32 || context < 1 || context > 8192 || repeats < 1)
            throw std::invalid_argument("benchmark bounds");
        constexpr int layers = 28, heads = 16, kvheads = 8, dim = 128;
        tts::Backend g;
        g.reserve_attention_workspace(batch * heads);
        tts::PagedKVCache pool(g.memory, layers, kvheads * dim, tts::Type::BF16, batch,
                               context, page);
        std::vector<int> lengths(batch), positions(batch);
        for (int i = 0; i < batch; ++i) {
            lengths[i] = std::max(1, context - i * 7);
            positions[i] = lengths[i] - 1;
        }
        pool.reserve(pool.required_pages(lengths));
        tts::PagedKVCache::Lease lease(pool, g.stream, lengths);
        auto view = pool.view();
        CUDA(cudaMemsetAsync(view.keys.data, 0, view.keys.size(), g.stream));
        CUDA(cudaMemsetAsync(view.values.data, 0, view.values.size(), g.stream));
        tts::Buffer q(size_t(batch) * heads * dim * 2, &g.memory), out(q.size(), &g.memory), pos(batch * sizeof(int), &g.memory);
        g.zero(q);
        g.upload(pos.data(), positions.data(), pos.size());
        tts::Graph graph;
        graph.capture(g.stream, [&] {
            for (int layer = 0; layer < layers; ++layer)
                g.attention(q.view(batch, heads * dim), pool.keys(layer), pool.values(layer),
                            out.view(batch, heads * dim), pos.ptr<int>(), batch, 1, 0, context,
                            heads, kvheads, dim, true, 0, -1, nullptr, pool.layout());
        });
        for (int i = 0; i < 5; ++i)
            graph.run(g.stream);
        g.sync();
        cudaEvent_t start, stop;
        CUDA(cudaEventCreate(&start));
        CUDA(cudaEventCreate(&stop));
        std::vector<float> times;
        for (int i = 0; i < repeats; ++i) {
            CUDA(cudaEventRecord(start, g.stream));
            graph.run(g.stream);
            CUDA(cudaEventRecord(stop, g.stream));
            CUDA(cudaEventSynchronize(stop));
            float ms;
            CUDA(cudaEventElapsedTime(&ms, start, stop));
            times.push_back(ms * 1000 / layers);
        }
        std::sort(times.begin(), times.end());
        std::cout << "{\"page\":" << page << ",\"batch\":" << batch << ",\"max_keys\":" << context
                  << ",\"kv_slab_bytes\":" << pool.storage_capacity_bytes()
                  << ",\"median_us_per_layer\":" << times[times.size() / 2]
                  << ",\"min_us\":" << times.front() << ",\"max_us\":" << times.back() << "}\n";
        CUDA(cudaEventDestroy(start));
        CUDA(cudaEventDestroy(stop));
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
