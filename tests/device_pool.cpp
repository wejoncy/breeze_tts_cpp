#include "runtime/device.h"
#include "runtime/graph.h"
#include "runtime/tensor_storage.h"
#include <cassert>
#include <iostream>
#include <type_traits>
#include <random>
using namespace tts;
static_assert(!std::is_constructible_v<Buffer, size_t>);
static_assert(!std::is_constructible_v<Tensor, TensorShape>);
int main() {
    // Churn arbitrary lifetimes and holes, then prove complete coalescing.
    BlockPool blocks(1 << 20);
    std::vector<std::pair<size_t, size_t>> live;
    std::mt19937 random(42);
    for (int i = 0; i < 10000; ++i) {
        if (!live.empty() && random() % 2) {
            auto index = random() % live.size();
            blocks.deallocate(live[index].first);
            live.erase(live.begin() + index);
        } else {
            size_t size = 1 + random() % 4096;
            try {
                auto offset = blocks.allocate(size);
                for (auto [start, count] : live)
                    assert(offset + size <= start || start + count <= offset);
                live.emplace_back(offset, size);
            } catch (const std::bad_alloc &) {
                assert(blocks.largest() < size);
            }
        }
    }
    for (auto [offset, size] : live)
        blocks.deallocate(offset);
    assert(blocks.available() == (1 << 20) && blocks.largest() == (1 << 20));
    DeviceContext first(0), second(0);
    constexpr size_t bytes = 1 << 20;
    {
        DeviceContext limited(0, 4 * bytes);
        Buffer persistent(2 * bytes, &limited);
        bool rejected = false;
        try {
            Buffer exceeds_budget(3 * bytes, &limited);
        } catch (const DeviceOutOfMemory &) {
            rejected = true;
        }
        assert(rejected && limited.available_bytes() == 2 * bytes);
        limited.reserve_runtime(2 * bytes);
        Buffer temporary(bytes, &limited);
        limited.sync();
        assert(limited.stats().used_bytes <= 4 * bytes);
        assert(limited.stats().peak_used_bytes <= 4 * bytes);
        assert(limited.available_bytes() == bytes);
    }
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
            ArenaLease lease(arena);
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
    first.reserve_runtime(4 * bytes);
    auto physical = first.stats();
    {
        Buffer a(bytes, &first), b(bytes, &first), c(bytes, &first);
        CUDA(cudaMemsetAsync(a.data(), 3, a.size(), first.stream));
        void *old = b.data();
        b = Buffer();
        Buffer reused(bytes, &first);
        assert(reused.data() == old);
        bool exhausted = false;
        try {
            Buffer too_large(2 * bytes, &first);
        } catch (const DeviceOutOfMemory &) {
            exhausted = true;
        }
        assert(exhausted);
        assert(first.available_bytes() == bytes);
    }
    assert(first.available_bytes() == 4 * bytes);
    Buffer whole(4 * bytes, &first);
    assert(first.stats().used_bytes == physical.used_bytes);
    assert(first.stats().reserved_bytes == physical.reserved_bytes);
    std::cout << "pool ownership, cross-pool moves, stream-ordered release, arena reuse, stats and "
                 "capture rejection passed\n";
}
