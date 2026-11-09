#include "tts/engine.h"
#include <chrono>
#include <future>
#include <iostream>
#include <map>
#include <thread>
using namespace std::chrono_literals;
static void require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}
template <class F> static void rejects(F &&operation) {
    bool rejected = false;
    try { operation(); } catch (const std::exception &) { rejected = true; }
    require(rejected, "expected rejection");
}
static tts::GenerationFrame pop(tts::Engine &engine) {
    auto &queue = engine.output_queue();
    std::unique_lock<std::mutex> lock(queue.mutex);
    require(queue.available.wait_for(lock, 10s, [&] { return queue.closed || !queue.frames.empty(); }),
            "output queue stalled");
    require(!queue.frames.empty(), "engine closed without terminal output");
    auto frame = std::move(queue.frames.front()); queue.frames.pop_front();
    for (const auto &chunk : frame.chunks) queue.bytes -= chunk.second.size() * sizeof(int16_t);
    return frame;
}
using Audio = std::map<int, std::vector<int16_t>>;
static Audio collect(tts::Engine &engine, const std::string &id) {
    Audio result;
    for (;;) {
        auto frame = pop(engine);
        require(frame.request_id == id, "unexpected connection output");
        if (frame.error) std::rethrow_exception(frame.error);
        require(!frame.cancelled, "unexpected cancellation");
        for (auto &chunk : frame.chunks) {
            auto &pcm = result[chunk.first]; pcm.insert(pcm.end(), chunk.second.begin(), chunk.second.end());
        }
        if (frame.done) return result;
    }
}
static tts::Request request(const std::string &id, const tts::GenerationOptions &options, size_t lanes = 1) {
    tts::Request job{id, {}, options};
    for (size_t lane = 0; lane < lanes; ++lane) {
        tts::Input input; input.text = "你好，今天我们沿着河边散步，看看小镇的风景。";
        job.inputs.push_back(std::move(input));
    }
    return job;
}
int main(int argc, char **argv) {
    try {
        if (argc != 2) throw std::invalid_argument("usage: tts-engine-tests MODEL_DIRECTORY");
        tts::Engine engine(argv[1], 0, 4, 2048, 128, tts::Quantization::INT8,
                           size_t(2048) << 20, size_t(12288) << 20);
        auto options = engine.default_options(); options.max_frames = 8;
        options.do_sample = options.depth_do_sample = false;
        options.temperature = options.depth_temperature = 0;
        rejects([&] { engine.add_request(request("before-start", options)); });
        engine.run_async(); engine.run_async(); engine.configure_serving(32, 1 << 20);
        engine.add_request(request("graph", options)); auto graph = collect(engine, "graph");
        require(!graph[0].empty(), "empty graph audio");
        auto eager = options; eager.use_cuda_graph = false;
        engine.add_request(request("eager", eager));
        require(collect(engine, "eager") == graph, "graph/eager PCM mismatch");
        auto snapshot = request("snapshot", options); engine.add_request(snapshot);
        snapshot.inputs.clear(); snapshot.options.max_frames = 0;
        require(collect(engine, "snapshot") == graph, "input queue borrowed caller storage");
        engine.configure_serving(1, 1 << 20); engine.add_request(request("permit", options));
        rejects([&] { engine.add_request(request("overloaded", options)); });
        collect(engine, "permit"); engine.add_request(request("permit", options)); collect(engine, "permit");
        engine.configure_serving(32, 1 << 20);
        std::vector<std::future<void>> producers;
        for (int index = 0; index < 9; ++index) producers.push_back(std::async(std::launch::async, [&, index] {
            auto job = request("connection:" + std::to_string(index), options);
            job.options.max_frames = 5 + index;
            job.inputs[0].text = "第" + std::to_string(index + 1) + "篇。" + job.inputs[0].text;
            engine.add_request(job);
        }));
        for (auto &producer : producers) producer.get();
        std::map<std::string, size_t> samples; size_t completed = 0;
        while (completed < 9) {
            auto frame = pop(engine); if (frame.error) std::rethrow_exception(frame.error);
            for (const auto &chunk : frame.chunks) samples[frame.request_id] += chunk.second.size();
            completed += frame.done;
        }
        for (int index = 0; index < 9; ++index)
            require(samples["connection:" + std::to_string(index)] == size_t(5 + index) * engine.audio_format().samples_per_frame,
                    "shared output queue lost or mixed connection audio");
        engine.add_request(request("multi", options, 9)); auto multi = collect(engine, "multi");
        require(multi.size() == 9, "lost resource-constrained input wave");
        for (const auto &lane : multi) require(lane.second.size() == graph[0].size(), "input wave lost PCM");
        auto invalid = request("invalid", options); invalid.options.max_frames = 0;
        engine.add_request(invalid); auto error = pop(engine);
        require(error.request_id == "invalid" && error.done && bool(error.error), "missing queued request error");
        auto long_options = options; long_options.max_frames = 300;
        engine.add_request(request("cancel", long_options)); require(!pop(engine).done, "cancel request did not start");
        auto memory = std::async(std::launch::async, [&] { return engine.memory_stats(); });
        require(memory.wait_for(500ms) == std::future_status::ready, "control starved behind generation"); memory.get();
        engine.cancel_request("cancel");
        for (;;) { auto frame = pop(engine); if (frame.done) { require(frame.cancelled, "missing cancellation"); break; } }
        rejects([&] { engine.decode_codes({-1}); });
        auto reference = engine.prepare_reference(std::vector<float>(24000, 0), "你好。");
        require(reference && !reference->codes.empty(), "reference encoding failed");
        require(!engine.decode_codes(reference->codes).empty(), "audio code decoding failed");
        std::vector<int> decode_codes;
        for (int repeat = 0; repeat < 8; ++repeat)
            decode_codes.insert(decode_codes.end(), reference->codes.begin(), reference->codes.end());
        const auto decoded = engine.decode_codes(decode_codes);
        auto interleaved_options = options; interleaved_options.max_frames = 32;
        engine.add_request(request("interleaved", interleaved_options));
        require(!pop(engine).done, "interleaved generation did not start");
        auto decoding = std::async(std::launch::async, [&] { return engine.decode_codes(decode_codes); });
        size_t frames_during_decode = 0;
        for (;;) {
            auto frame = pop(engine);
            require(frame.request_id == "interleaved" && !frame.error, "interleaved generation failed");
            if (!frame.chunks.empty() && decoding.wait_for(0ms) != std::future_status::ready)
                ++frames_during_decode;
            if (frame.done) break;
        }
        require(frames_during_decode >= 4, "code decoding monopolized Runner");
        require(decoding.wait_for(2s) == std::future_status::ready, "generation starved code decoding");
        require(decoding.get() == decoded, "interleaving corrupted audio history or positions");
        std::cout << "generation frames during code decoding=" << frames_during_decode << '\n';
        tts::Stats sink_stats;
        auto caller = std::this_thread::get_id();
        require(engine.generate(request("unused", options).inputs, options,
            [&](int, const int16_t *, size_t) {
                require(std::this_thread::get_id() == caller, "audio sink crossed threads");
                return false;
            }, sink_stats) == 1, "sink cancellation was ignored");
        rejects([&] { engine.generate(request("unused", options).inputs, options,
            [](int, const int16_t *, size_t) -> bool { throw std::runtime_error("sink failed"); }, sink_stats); });
        engine.add_request(request("after-sink", options));
        require(collect(engine, "after-sink") == graph, "sink cancellation left stale FIFO frames");
        auto sampled = options; sampled.do_sample = sampled.depth_do_sample = true;
        sampled.temperature = sampled.depth_temperature = .9f;
        engine.add_request(request("seed", sampled)); auto first = collect(engine, "seed");
        engine.add_request(request("seed", sampled)); require(collect(engine, "seed") == first, "cache RNG reuse mismatch");
        engine.configure_serving(32, 1); auto one = options; one.max_frames = 1;
        engine.add_request(request("overflow", one)); error = pop(engine);
        require(error.done && bool(error.error), "shared output overflow did not terminate request");
        engine.configure_serving(32, 1 << 20);
        engine.add_request(request("after-overflow", options)); require(collect(engine, "after-overflow") == graph, "overflow affected neighbor");
        engine.add_request(request("shutdown", long_options)); engine.shutdown();
        require(engine.output_queue().closed, "shutdown did not close shared output queue");
        engine.shutdown(); rejects([&] { engine.run_async(); });
        uint64_t hash = 1469598103934665603ULL;
        for (auto sample : graph[0]) { hash = (hash ^ uint8_t(sample)) * 1099511628211ULL;
            hash = (hash ^ uint8_t(uint16_t(sample) >> 8)) * 1099511628211ULL; }
        std::cout << "shared FIFO engine checks passed; greedy_pcm_hash=" << hash << '\n';
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
