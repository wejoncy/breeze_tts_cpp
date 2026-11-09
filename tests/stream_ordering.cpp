#include "runtime/device.h"
#include "execution/output_sampler.h"
#include "runtime/kv_cache.h"
#include <chrono>
#include <thread>
#include <iostream>

static void require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}
int main() {
    try {
        tts::DeviceContext memory(0, size_t(64) << 20);
        tts::Tensor::Scope allocation(memory);
        tts::PagedKVCache kv(memory, 1, 32, tts::Type::F32, 4, 256, 64);
        tts::Buffer input(sizeof(int)), output(sizeof(int));
        memory.reserve_transfers(1024); // Wrap repeatedly with one successor upload in flight.
        memory.reserve_runtime(size_t(1) << 20);
        require(memory.stream != memory.copy_stream, "compute and transfer streams are identical");
        tts::OutputSampler sampler(memory, sizeof(int));
        auto collect = [&] {
            auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!sampler.ready()) {
                require(std::chrono::steady_clock::now() < deadline, "cross-stream event dependency deadlocked");
                std::this_thread::yield();
            }
            return sampler.collect().values.front().value.ptr<int>()[0];
        };
        int source = 1;
        input.copy_({&source, 1, 1, tts::Type::I32}, true);
        source = -1;
        for (int expected = 1; expected <= 128; ++expected) {
            output.view(1, 1, tts::Type::I32).copy_(input.view(1, 1, tts::Type::I32), true);
            sampler.sample({{{"value", output.view(1, 1, tts::Type::I32)}}});
            // The predecessor event must retire only its upload prefix while
            // the successor still owns a snapshot in the same fixed ring.
            if (expected < 128) {
                source = expected + 1;
                input.copy_({&source, 1, 1, tts::Type::I32}, true);
                source = -1;
            }
            // Immutable KV pointers use independent persistent pinned storage.
            if (expected <= 16) kv.reserve(expected);
            require(collect() == expected, "compute/copy event order corrupted a sampled result");
        }
        bool rejected = false;
        try {
            sampler.sample({{{"value", output.view(1, 1, tts::Type::I32)},
                             {"invalid", {nullptr, 1, 1, tts::Type::I32}}}});
        } catch (const std::logic_error &) { rejected = true; }
        require(rejected, "invalid partial readback was accepted");
        memory.sync();
        sampler.discard_after_drain();
        sampler.sample({{{"value", output.view(1, 1, tts::Type::I32)}}});
        require(collect() == 128, "partial-readback cleanup poisoned subsequent sampling");
        std::cout << "two-stream ordering, snapshot lifetime and partial-readback checks passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
