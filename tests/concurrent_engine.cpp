#include "tts/engine.h"
#include "models/breeze/model.h"
#include "execution/model_runner.h"
#include "execution/request_planner.h"
#include "execution/prefill.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <type_traits>
static void check_packed_prefill(const tts::Model &model, tts::Backend &gpu) {
    tts::RequestPlanner planner(model.tokenizer, model.config, 4, 256);
    auto reference =
        std::make_shared<tts::Reference>("Hello.", std::vector<int>(model.config.codebooks, 0),
                                         model.config.codebooks, model.config.codec.codebook_size);
    std::vector<std::unique_ptr<tts::RequestPlan>> plans;
    std::vector<const tts::PreparedBatch *> inputs;
    std::string long_text;
    for (int i = 0; i < 40; ++i)
        long_text += "Welcome aboard. ";
    for (int i = 0; i < 4; ++i) {
        tts::GenerationOptions options;
        options.max_frames = 4;
        options.dual_cfg = i == 0;
        options.cfg_scale = i == 1 ? 2 : 1;
        plans.push_back(planner.prepare(
            {{i == 3   ? "Hi."
              : i == 1 ? long_text
              : i == 2 ? "An independent request with a different prompt length and voice."
                       : "Welcome aboard.",
              i == 3 ? "" : "Calm.", i == 3 ? "" : "S0", i == 0 ? reference : nullptr}},
            options));
        inputs.push_back(&plans.back()->inputs.front());
    }
    auto packed = tts::pack_prefill(inputs);
    assert(packed.text_rows > 256 && packed.max_prompt < 256);
    gpu.reserve_attention_workspace(size_t(packed.lanes) *
                                    std::max(packed.max_prompt, 2 * packed.max_text) *
                                    std::max(model.text.config.heads, model.backbone.config.heads));
    model.text.prepare_range(gpu, packed.text_rows);
    model.backbone.prepare_range(gpu, packed.storage_rows);
    auto run = [&](const tts::PreparedBatch &input) {
        tts::DeviceArena arena(gpu.memory, tts::prefill_workspace_bytes(model, input, {}));
        tts::Buffer last(size_t(input.lanes) * model.backbone.config.hidden * 2, &gpu.memory);
        tts::Prefill prefill(model, gpu, arena, input, {},
                             last.view(input.lanes, model.backbone.config.hidden));
        while (!prefill.step()) {
        }
        std::vector<uint16_t> result(last.size() / 2);
        gpu.download(result.data(), last.data(), last.size());
        return result;
    };
    std::vector<uint16_t> expected;
    for (const auto *input : inputs) {
        auto output = run(*input);
        expected.insert(expected.end(), output.begin(), output.end());
    }
    const auto actual = run(packed);
    double error = 0, norm = 0;
    auto value = [](uint16_t bits) {
        uint32_t word = uint32_t(bits) << 16;
        float result;
        std::memcpy(&result, &word, sizeof(result));
        return result;
    };
    for (size_t i = 0; i < actual.size(); ++i) {
        assert(std::isfinite(value(actual[i])));
        const double difference = value(actual[i]) - value(expected[i]);
        error += difference * difference;
        norm += double(value(expected[i])) * value(expected[i]);
    }
    const double rrmse = std::sqrt(error / norm);
    double worst_lane = 0;
    for (int lane = 0; lane < packed.lanes; ++lane) {
        double lane_error = 0, lane_norm = 0;
        for (int col = 0; col < model.backbone.config.hidden; ++col) {
            size_t i = size_t(lane) * model.backbone.config.hidden + col;
            double delta = value(actual[i]) - value(expected[i]);
            lane_error += delta * delta;
            lane_norm += double(value(expected[i])) * value(expected[i]);
        }
        worst_lane = std::max(worst_lane, std::sqrt(lane_error / lane_norm));
    }
    std::cout << "packed vs independent prefill relative RMS error=" << rrmse
              << " worst_lane=" << worst_lane << std::endl;
    // Record full-model quantization drift; quantized GEMMs have independent
    // dot-product oracles in quantization.cu. Request isolation must remain
    // bit-identical below, regardless of selected geometry or precision.
    if (model.backbone.config.quantization == tts::Quantization::None) {
        assert(rrmse < .01);
        assert(worst_lane < .01);
    }
    // Same shapes, different unrelated requests: the first request must be bit-identical.
    for (int row = inputs.front()->text_rows; row < packed.text_rows; ++row)
        packed.tokens[row] = (packed.tokens[row] + 1) % model.config.text_vocab;
    const auto changed = run(packed);
    assert(std::equal(actual.begin(),
                      actual.begin() + inputs.front()->lanes * model.backbone.config.hidden,
                      changed.begin()));
}
static void check_layer_checkpoints(const tts::Model &model, tts::Backend &gpu) {
    for (const auto *transformer : {&model.text, &model.backbone, &model.depth}) {
        const auto &c = transformer->config;
        int rows = std::min(7, c.max_positions);
        tts::TransformerState full(c, 1, rows, rows, &gpu.memory);
        tts::TransformerState stepped(c, 1, rows, rows, &gpu.memory);
        tts::Buffer a(size_t(rows) * c.hidden * 2, &gpu.memory), b(a.size(), &gpu.memory);
        std::vector<uint16_t> input(a.size() / 2), expected(input.size()), actual(input.size());
        for (size_t i = 0; i < input.size(); ++i)
            input[i] = uint16_t(0x3d00 + (i * 37) % 0x400) | (i % 3 ? 0 : 0x8000);
        gpu.upload(a.data(), input.data(), a.size());
        gpu.upload(b.data(), input.data(), b.size());
        transformer->forward(gpu, full, a.view(rows, c.hidden), nullptr, 1, rows, 0, 0);
        for (int layer = 0; layer < c.layers; ++layer)
            transformer->forward(gpu, stepped, b.view(rows, c.hidden), nullptr, 1, rows, 0, 0, {},
                                 nullptr, nullptr, {layer, layer + 1});
        gpu.download(expected.data(), a.data(), a.size());
        gpu.download(actual.data(), b.data(), b.size());
        assert(actual == expected);
    }
    std::cout << "all transformer layer checkpoints match full forward bit-for-bit\n";
}
static void check_padded_replay(tts::Model &model, tts::ModelRunner &runner) {
    tts::RequestPlanner planner(model.tokenizer, model.config, 4, 256);
    std::vector<std::vector<int16_t>> expected;
    auto reference =
        std::make_shared<tts::Reference>("Hello.", std::vector<int>(model.config.codebooks, 0),
                                         model.config.codebooks, model.config.codec.codebook_size);
    for (bool graph : {false, true}) {
        std::vector<std::unique_ptr<tts::RequestPlan>> plans;
        std::vector<std::unique_ptr<tts::ModelRunner::RequestAdmission>> admissions;
        std::vector<std::unique_ptr<tts::ModelRunner::RequestState>> states;
        std::vector<tts::ModelRunner::RequestState *> rows;
        std::vector<tts::AudioCallback> callbacks;
        std::vector<std::vector<int16_t>> audio(4);
        for (int i = 0; i < 4; ++i) {
            tts::GenerationOptions options;
            options.max_frames = 16;
            options.do_sample = options.depth_do_sample = false;
            options.use_cuda_graph = graph;
            options.dual_cfg = i < 3;
            options.cfg_scale = i == 3 ? 2 : 1;
            plans.push_back(
                planner.prepare({{"Welcome aboard.", "Calm.", "S0", reference}}, options));
            admissions.push_back(runner.admit(*plans.back()));
            states.push_back(runner.prepare(*plans.back(), 0, *admissions.back()));
            rows.push_back(states.back().get());
            callbacks.push_back([&, i](int, const int16_t *pcm, size_t count) {
                audio[i].insert(audio[i].end(), pcm, pcm + count);
                return true;
            });
        }
        while (std::any_of(rows.begin(), rows.end(), [](auto *row) { return !row->prefilled(); }))
            assert(runner.prefill_step(rows) == rows.size());
        // All four requests start together: identical membership gives eleven
        // real CFG lanes and one padded lane in both eager and graph execution.
        while (std::any_of(rows.begin(), rows.end(), [](auto *row) { return !row->done(); }))
            runner.step(rows, callbacks);
        if (!graph)
            expected = std::move(audio);
        else
            assert(audio == expected);
    }
}
static void check_prefill_release(tts::Model &model, tts::Backend &gpu, tts::ModelRunner &runner) {
    tts::RequestPlanner planner(model.tokenizer, model.config, 4, 256);
    tts::GenerationOptions options;
    options.max_frames = 2;
    options.do_sample = options.depth_do_sample = false;
    std::vector<std::unique_ptr<tts::RequestPlan>> plans;
    std::vector<std::unique_ptr<tts::ModelRunner::RequestAdmission>> admissions;
    std::vector<std::unique_ptr<tts::ModelRunner::RequestState>> states;
    std::vector<tts::ModelRunner::RequestState *> rows;
    size_t single_workspace = 0;
    std::string text;
    for (int i = 0; i < 30; ++i)
        text += "Welcome aboard. ";
    for (int i = 0; i < 4; ++i) {
        plans.push_back(planner.prepare({{text}}, options));
        const auto &input = plans.back()->inputs.front();
        single_workspace =
            std::max(single_workspace, tts::prefill_workspace_bytes(model, input, {}) + 4096);
        admissions.push_back(runner.admit(*plans.back()));
        states.push_back(runner.prepare(*plans.back(), 0, *admissions.back()));
        rows.push_back(states.back().get());
    }
    assert(runner.prefill_step(rows) == 4); // Retain a layer checkpoint.
    rows.erase(rows.begin() + 1);
    states[1].reset(); // Releasing one member invalidates the borrowed batch safely.
    // Simulate pressure using the existing pool: each input still fits alone.
    const auto available = gpu.memory.largest_available_bytes();
    assert(available > single_workspace);
    tts::Buffer pressure(available - single_workspace, &gpu.memory);
    while (std::any_of(rows.begin(), rows.end(), [](auto *row) { return !row->prefilled(); })) {
        auto advanced = runner.prefill_step(rows);
        assert(advanced > 0 && advanced < rows.size());
    }
    runner.step(rows, std::vector<tts::AudioCallback>(
                          rows.size(), [](int, const int16_t *, size_t) { return true; }));
    std::cout << "packed prefill cancellation and constrained-pool progress passed\n";
}
static void check_first_frame_eos(const std::string &directory, tts::Quantization quantization) {
    tts::Backend gpu(0, quantization);
    tts::Model model(gpu, directory, tts::load_model_config(directory, 4, 256, quantization));
    check_packed_prefill(model, gpu);
    // Two opposite finite heads guarantee EOS on one sign of the same hidden state.
    // Only this test's device buffer changes; the checkpoint is untouched.
    std::vector<uint16_t> row(model.lm_head.cols), head(model.lm_head.numel());
    gpu.download(row.data(), model.lm_head.data, row.size() * sizeof(uint16_t));
    tts::Buffer weights(model.lm_head.size(), &gpu.memory);
    CUDA(cudaMemcpyAsync(weights.data(), model.lm_head.data, weights.size(),
                         cudaMemcpyDeviceToDevice, gpu.stream));
    model.lm_head = weights.view(model.lm_head.rows, model.lm_head.cols);
    tts::ModelRunner runner(model, gpu, 4, 256, 128);
    check_layer_checkpoints(model, gpu);
    tts::RequestPlanner planner(model.tokenizer, model.config, 4, 256);
    check_padded_replay(model, runner);
    check_prefill_release(model, gpu, runner);
    tts::GenerationOptions options;
    options.max_frames = 2;
    options.do_sample = options.depth_do_sample = false;
    bool ended_without_audio = false;
    for (int sign = 0; sign < 2; ++sign) {
        for (int r = 0; r < model.lm_head.rows; ++r)
            for (int c = 0; c < model.lm_head.cols; ++c)
                head[size_t(r) * row.size() + c] =
                    row[c] ^ ((sign ^ (r == model.config.backbone_eos)) ? 0x8000 : 0);
        gpu.upload(weights.data(), head.data(), weights.size());
        auto plan = planner.prepare({{"Hello."}}, options);
        auto admission = runner.admit(*plan);
        auto state = runner.prepare(*plan, 0, *admission);
        while (!state->prefilled() && !state->done())
            runner.prefill_step({state.get()});
        assert(state->prefilled());
        bool emitted = false;
        runner.step({state.get()}, {[&](int, const int16_t *, size_t) {
                        emitted = true;
                        return true;
                    }});
        if (!emitted) {
            assert(state->done() && state->stats().frames == 0);
            ended_without_audio = true;
        }
    }
    assert(ended_without_audio);
}
int main(int argc, char **argv) {
    if (argc < 2)
        return 2;
    auto mode = argc > 2 ? std::string(argv[2]) : "";
    {
        tts::Engine engine(argv[1], 0, 3, 512, 128,
                           mode == "int8"  ? tts::Quantization::INT8
                           : mode == "fp8" ? tts::Quantization::FP8
                                           : tts::Quantization::None);
        tts::GenerationOptions options;
        options.max_frames = 24;
        options.do_sample = options.depth_do_sample = false;
        tts::Request long_request{
            "long",
            {{"This is a longer request for testing continuous speech generation."}},
            options};
        static_assert(std::is_void_v<decltype(engine.add_request(long_request))>);
        bool not_started = false;
        try {
            engine.add_request(long_request);
        } catch (const std::runtime_error &error) {
            not_started = std::string(error.what()).find("run_async") != std::string::npos;
        }
        assert(not_started);
        engine.run_async();
        engine.run_async();
        engine.add_request(long_request);
        // Submission snapshots input, identity and options.
        long_request.request_id = "changed";
        long_request.inputs.clear();
        assert(!engine.read_result("long").chunks.empty());
        options.max_frames = 3;
        engine.add_request({"short", {{"你好。"}}, options});
        engine.add_request({"short", {{"Duplicate."}}, options});
        bool duplicate = false;
        try {
            engine.read_result("short");
        } catch (const std::invalid_argument &) {
            duplicate = true;
        }
        assert(duplicate);
        bool merged = false;
        size_t samples = 0;
        while (true) {
            auto frame = engine.read_result("short");
            assert(frame.request_id == "short");
            merged |= frame.batch_size > 1;
            for (const auto &chunk : frame.chunks)
                samples += chunk.second.size();
            if (frame.done)
                break;
        }
        assert(merged);
        assert(samples == size_t(3 * engine.audio_format().samples_per_frame));
        // A terminal read consumes the ID; cancellation discards it immediately.
        engine.cancel_request("short");
        engine.cancel_request("long");
        engine.cancel_request("long"); // Repeated cancellation is an idempotent event.
        bool removed = false;
        try {
            engine.read_result("long");
        } catch (const std::invalid_argument &) {
            removed = true;
        }
        assert(removed);
        engine.add_request({"short", {{"Hello."}}, options});
        size_t reused_samples = 0;
        while (true) {
            auto frame = engine.read_result("short");
            for (const auto &chunk : frame.chunks)
                reused_samples += chunk.second.size();
            if (frame.done)
                break;
        }
        assert(reused_samples == samples);
        // Validate every lane before emitting any part of a larger-than-capacity request.
        std::vector<tts::Input> invalid_batch(4, {"Hello."});
        invalid_batch.back().text = std::string(4096, 'x');
        engine.add_request({"bad-batch", invalid_batch, options});
        bool invalid = false;
        try {
            engine.read_result("bad-batch");
        } catch (const std::invalid_argument &) {
            invalid = true;
        }
        assert(invalid);
        // Unread completed output remains bounded by the request admission limit.
        engine.configure_serving(1, 1 << 20);
        engine.add_request({"unread", {{"Hello."}}, options});
        bool full = false;
        try {
            engine.add_request({"overflow", {{"Hello."}}, options});
        } catch (const std::runtime_error &) {
            full = true;
        }
        assert(full);
        // The owner stops ingress, then shuts down while requests remain live.
        engine.shutdown();
        engine.shutdown();
        bool stopped = false;
        try {
            engine.add_request({"closed", {{"Hello."}}, options});
        } catch (const std::runtime_error &) {
            stopped = true;
        }
        assert(stopped);
        stopped = false;
        try {
            engine.run_async();
        } catch (const std::runtime_error &) {
            stopped = true;
        }
        assert(stopped);
    }
    check_first_frame_eos(argv[1], mode == "int8"  ? tts::Quantization::INT8
                                   : mode == "fp8" ? tts::Quantization::FP8
                                                   : tts::Quantization::None);
    std::cout << "native IDs: asynchronous submission, shared batching, duplicate rejection, "
                 "cancellation, reuse and shutdown OK\n";
}
