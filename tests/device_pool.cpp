#include "device.h"
#include "graph.h"
#include "tensor_storage.h"
#include <cassert>
#include <iostream>
#include <type_traits>
using namespace tts;
static_assert(!std::is_constructible_v<Buffer, size_t>);
static_assert(!std::is_constructible_v<Tensor, TensorShape>);
int main() {
    DeviceContext first(0), second(0);
    constexpr size_t bytes = 1 << 20;
    {
        Buffer a(bytes, &first), b(2 * bytes, &second);
        CUDA(cudaMemsetAsync(a.data(), 3, a.size(), first.stream));
        CUDA(cudaMemsetAsync(b.data(), 7, b.size(), second.stream));
        a = std::move(b); // Free through the old pool; retain the new allocator.
        first.sync();
        second.sync();
        assert(first.stats().used_bytes == 0);
        assert(second.stats().used_bytes >= 2 * bytes);
        assert(b.data() == nullptr && b.size() == 0);
        unsigned char value = 0;
        CUDA(cudaMemcpyAsync(&value, a.data(), 1, cudaMemcpyDeviceToHost, second.stream));
        second.sync();
        assert(value == 7);
    }
    second.trim();
    assert(second.stats().used_bytes == 0);
    assert(second.stats().peak_used_bytes >= 2 * bytes);
    {
        DeviceArena arena(first, bytes);
        auto reserved = first.stats().used_bytes;
        {
            ArenaLease lease(arena, first.stream);
            Buffer a(100, &arena), b(100, &arena);
            assert(static_cast<char *>(b.data()) - static_cast<char *>(a.data()) == 256);
            assert(first.stats().used_bytes == reserved);
        }
        arena.reserve(2 * bytes);
        assert(arena.capacity() == 2 * bytes);
    }
    first.trim();
    assert(first.stats().used_bytes == 0);
    WorkspacePlan plan;
    Buffer metadata(bytes, &plan);
    assert(metadata.size() == bytes && metadata.data() == nullptr);
    assert(first.stats().used_bytes == 0);
    Graph graph;
    bool rejected = false;
    try {
        graph.capture(first.stream, [&] { Buffer forbidden(bytes, &first); });
    } catch (const std::logic_error &) {
        rejected = true;
    }
    assert(rejected);
    cudaStreamCaptureStatus status;
    CUDA(cudaStreamIsCapturing(first.stream, &status));
    assert(status == cudaStreamCaptureStatusNone);
    first.sync();
    assert(first.stats().used_bytes == 0);
    CUDA(cudaGetLastError());
    std::cout << "pool ownership, cross-pool moves, stream-ordered release, arena reuse, stats and "
                 "capture rejection passed\n";
}
