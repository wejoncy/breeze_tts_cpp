#include "tts/engine.h"
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>

static void require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}
static size_t collect(tts::Engine &engine, const std::string &id) {
    size_t samples = 0;
    for (;;) {
        auto &queue = engine.output_queue();
        std::unique_lock<std::mutex> lock(queue.mutex);
        require(queue.available.wait_for(lock, std::chrono::seconds(10), [&] { return !queue.frames.empty(); }), "output stalled");
        auto frame = std::move(queue.frames.front()); queue.frames.pop_front();
        for (const auto &chunk : frame.chunks) queue.bytes -= chunk.second.size() * sizeof(int16_t);
        lock.unlock();
        if (frame.error) std::rethrow_exception(frame.error);
        require(frame.request_id == id && !frame.cancelled, "invalid preparation result");
        for (auto &chunk : frame.chunks) samples += chunk.second.size();
        if (frame.done) return samples;
    }
}
int main(int argc, char **argv) {
    try {
        if (argc != 4) throw std::invalid_argument("usage: tts-preparation-tests MODEL_DIRECTORY REFERENCE_F32 TRANSCRIPT_TXT");
        std::ifstream audio(argv[2], std::ios::binary | std::ios::ate);
        auto bytes = audio.tellg();
        require(audio && bytes > 0 && size_t(bytes) % sizeof(float) == 0, "invalid F32 fixture");
        std::vector<float> mono(size_t(bytes) / sizeof(float));
        audio.seekg(0); audio.read(reinterpret_cast<char *>(mono.data()), bytes);
        std::ifstream text(argv[3]);
        std::string transcript{std::istreambuf_iterator<char>(text), {}};
        tts::Engine engine(argv[1], 0, 4, 512, 128, tts::Quantization::None,
                           size_t(2048) << 20, size_t(12288) << 20);
        engine.run_async();
        auto reference = engine.prepare_reference(mono, transcript, "zh");
        require(reference && reference->features, "missing reference features");
        auto options = engine.default_options();
        options.max_frames = 256;
        options.do_sample = false;
        auto job = [&](const std::string &id, bool english = false) {
            tts::Input input;
            input.text = english ? "Today we walk beside the river and enjoy the quiet afternoon."
                                 : "今天我们沿着河边散步，看看小镇的风景，再去公园听听鸟儿的歌声。";
            input.language = english ? "en" : "zh";
            input.reference = reference;
            return tts::Request{id, {input}, options};
        };
        // Every request needs text conditioning. More requests than GPU rows
        // exercise conditioning/generation interleaving and independent retirement.
        for (int i = 0; i < 9; ++i) engine.add_request(job("prep:" + std::to_string(i), i % 3 == 0));
        size_t completed = 0, total = 0;
        while (completed < 9) {
            auto &queue = engine.output_queue();
            std::unique_lock<std::mutex> lock(queue.mutex);
            require(queue.available.wait_for(lock, std::chrono::seconds(10), [&] { return !queue.frames.empty(); }), "conditioning stalled");
            auto frame = std::move(queue.frames.front()); queue.frames.pop_front();
            for (const auto &chunk : frame.chunks) { queue.bytes -= chunk.second.size() * sizeof(int16_t); total += chunk.second.size(); }
            if (frame.error) std::rethrow_exception(frame.error);
            completed += frame.done;
        }
        require(total > 0, "missing conditioned audio");
        auto multi = job("segments");
        multi.inputs[0].text_split_method = "cut5";
        multi.inputs[0].text = "今天我们沿着河边散步。小镇的风景十分美丽。傍晚再去公园听鸟儿唱歌。";
        multi.inputs.push_back(job("unused", true).inputs[0]);
        engine.add_request(multi);
        require(collect(engine, "segments") > 0, "segmented preparation did not complete");
        engine.add_request(job("cancel-prep"));
        engine.cancel_request("cancel-prep");
        { auto &queue = engine.output_queue(); std::unique_lock<std::mutex> lock(queue.mutex);
          queue.available.wait(lock, [&] { return !queue.frames.empty(); });
          for (const auto &chunk : queue.frames.front().chunks) queue.bytes -= chunk.second.size() * sizeof(int16_t);
          queue.frames.pop_front(); }
        engine.add_request(job("cancel-prep", true));
        require(collect(engine, "cancel-prep") > 0, "cancelled preparation corrupted reused ID");
        auto invalid = job("invalid-language");
        invalid.inputs[0].language = "not-a-language";
        engine.add_request(invalid);
        bool failed = false;
        try { collect(engine, "invalid-language"); } catch (const std::exception &) { failed = true; }
        require(failed, "invalid language was accepted");
        engine.add_request(job("after-error"));
        require(collect(engine, "after-error") > 0, "preparation error poisoned engine");
        engine.shutdown();
        std::cout << "GPU text preparation checks passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
