#include "runtime/execution_resources.h"
#include "runtime/graph.h"
#include <cassert>
#include <iostream>
using namespace tts;
int main() {
    DeviceContext memory(0);
    auto stream = memory.stream;
    {
        ExecutionResources<int, Graph> resources(memory);
        auto capture = [&] {
            auto workspace = resources.lease_workspace();
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
        resources.prepare(1024);
        capture();
        resources.prepare(1024);
        assert(resources.graphs_for(0).executable);
        capture();
        resources.prepare(8192); // workspace growth invalidates borrowed addresses
        assert(!resources.graphs_for(0).executable);
        capture();
        assert(!resources.graphs_for(1).executable);
        for (int key = 2; key < 32; ++key)
            resources.graphs_for(key);
        assert(resources.graphs_for(0).executable); // no eight-shape eviction
        resources.seal();
        assert(resources.find_graphs(0)->executable);
        assert(!resources.find_graphs(99));
        bool rejected = false;
        try {
            resources.graphs_for(99);
        } catch (const std::logic_error &) {
            rejected = true;
        }
        assert(rejected);
        rejected = false;
        try {
            resources.prepare(16384);
        } catch (const std::logic_error &) {
            rejected = true;
        }
        assert(rejected);
    }
    std::cout << "execution resources: reuse, workspace invalidation, plan revision passed\n";
}
