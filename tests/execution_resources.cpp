#include "execution_resources.h"
#include "graph.h"
#include <cassert>
#include <iostream>
using namespace tts;
int main() {
    DeviceContext memory(0);
    auto stream = memory.stream;
    {
        ExecutionResources<int, Graph> resources(memory, 2, 32, Type::BF16, 2, 256, 64);
        auto capture = [&] {
            auto workspace = resources.lease_workspace();
            auto kv = resources.lease_kv({32});
            Buffer buffer(1024, &resources.workspace());
            auto &graph = resources.graphs_for(0);
            if (!graph.executable)
                graph.capture(stream, [&] {
                    CUDA(cudaMemsetAsync(buffer.data(), 7, buffer.size(), stream));
                });
            graph.run(stream);
            unsigned char value = 0;
            CUDA(cudaMemcpyAsync(&value, buffer.data(), 1, cudaMemcpyDeviceToHost, stream));
            CUDA(cudaStreamSynchronize(stream));
            assert(value == 7);
        };
        resources.prepare(1024, {32});
        capture();
        resources.prepare(1024, {32});
        assert(resources.graphs_for(0).executable);
        capture();
        resources.prepare(8192, {32}); // workspace growth invalidates borrowed addresses
        assert(!resources.graphs_for(0).executable);
        capture();
        resources.prepare(8192, {129}); // KV growth must invalidate even without workspace growth
        assert(!resources.graphs_for(0).executable);
        capture();
        assert(!resources.graphs_for(1).executable);
    }
    std::cout << "execution resources: reuse, workspace/KV invalidation, plan revision passed\n";
}
