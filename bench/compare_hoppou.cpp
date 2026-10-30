// Benchmark adapter only: links the unmodified HoppouAI core and the native engine.
// argv: ENGINE(native|hoppou) MODEL PLAN.json RESULTS.json
#include "tts/engine.h"
#include "io/tokenizer.h"
#include "backends/cuda/backend.h"
#include "breeze/audio.h"
#include "breeze/generation.h"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
static double since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
int main(int argc, char **argv) {
    if (argc != 5) {
        std::cerr << "usage: compare_hoppou native|hoppou MODEL PLAN.json RESULTS.json\n";
        return 2;
    }
    try {
        const bool native = std::string(argv[1]) == "native";
        if (!native && std::string(argv[1]) != "hoppou")
            throw std::invalid_argument("engine must be native or hoppou");
        std::ifstream input(argv[3]);
        Json plan;
        input >> plan;
        auto load_start = Clock::now();
        std::unique_ptr<tts::Engine> engine;
        breeze::BreezeModel other;
        breeze::MimiCodec codec;
        if (native)
            engine = std::make_unique<tts::Engine>(argv[2], 0, 8, 2048);
        else {
            if (!other.load(argv[2], true) || !other.backend.is_gpu)
                throw std::runtime_error(
                    "baseline must load successfully on GPU; CPU fallback rejected");
            codec.init(other);
            if (other.cfg.sample_rate != 24000 || other.cfg.samples_per_frame != 1920 ||
                std::abs(other.cfg.depth_temperature - .9f) > 1e-6f ||
                other.cfg.depth_top_k != 50 || other.cfg.depth_top_p != 1.f)
                throw std::runtime_error("baseline metadata does not match benchmark settings");
        }
        Json result = {{"engine", argv[1]},
                       {"model", argv[2]},
                       {"backend", native ? "CUDA" : other.backend.name()},
                       {"load_ms", since(load_start)},
                       {"cases", Json::array()}};
        std::cout << result.dump() << std::endl;
        tts::Tokenizer token_audit(plan.at("checkpoint").get<std::string>() + "/tokenizer.json");
        result["prompt_token_counts"] = Json::array();
        for (const auto &item : plan.at("cases"))
            for (const auto &r : item.at("requests")) {
                std::string prompt = "[S0]<ins_bos>" + r.at("instruction").get<std::string>() +
                                     "<ins_eos>" + r.at("text").get<std::string>();
                auto tokens = token_audit.encode(prompt);
                if (!native && tokens != other.tok.encode(prompt, true))
                    throw std::runtime_error("tokenization differs for a comparison prompt");
                result["prompt_token_counts"].push_back(tokens.size());
            }
        std::vector<float> reference_audio;
        if (!breeze::read_wav(plan.at("reference_audio"), 24000, reference_audio))
            throw std::runtime_error("cannot read common reference fixture");
        auto ref_start = Clock::now();
        std::vector<int> reference_codes;
        int reference_frames = 0;
        std::shared_ptr<const tts::Reference> reference;
        const std::string transcript = plan.at("reference_text");
        if (native) {
            reference_codes = engine->encode_audio(reference_audio);
            reference_frames = reference_codes.size() / engine->audio_format().codebooks;
            reference = std::make_shared<const tts::Reference>(
                tts::Reference{transcript, reference_codes, engine->audio_format().codebooks,
                               engine->audio_format().codebook_size});
        } else
            reference_codes = codec.encode(reference_audio, reference_frames);
        result["reference_prepare_ms"] = since(ref_start);
        result["reference_frames"] = reference_frames;
        for (const auto &item : plan.at("cases")) {
            const int batch = item.at("requests").size(), frames = item.at("frames");
            const float cfg = item.value("cfg_scale", 1.f);
            const bool cloning = item.value("reference", false);
            std::vector<tts::Request> requests;
            for (const auto &r : item.at("requests"))
                requests.push_back(
                    {r.at("text"), r.at("instruction"), "S0", cloning ? reference : nullptr});
            Json observations = Json::array();
            for (int iteration = 0; iteration < plan.at("repeats").get<int>(); ++iteration) {
                std::vector<size_t> samples(batch), chunks(batch), first_size(batch);
                std::vector<double> first(batch, -1), four_frames(batch, -1), completed(batch, -1);
                std::vector<std::vector<float>> audio(batch);
                for (auto &lane : audio)
                    lane.reserve(size_t(frames) * 1920);
                tts::Range measurement("comparison.request");
                auto start = Clock::now();
                auto record = [&](int lane, size_t count) {
                    double now = since(start);
                    if (first[lane] < 0) {
                        first[lane] = now;
                        first_size[lane] = count;
                    }
                    samples[lane] += count;
                    ++chunks[lane];
                    if (samples[lane] >= 4 * 1920 && four_frames[lane] < 0)
                        four_frames[lane] = now;
                    completed[lane] = now;
                };
                Json stages = Json::array();
                if (native) {
                    tts::GenerationOptions opt;
                    opt.max_frames = frames;
                    opt.cfg_scale = cfg;
                    opt.seed = 42;
                    tts::Stats stats;
                    engine->generate(
                        requests, opt,
                        [&](int lane, const int16_t *pcm, size_t n) {
                            record(lane, n);
                            for (size_t j = 0; j < n; ++j)
                                audio[lane].push_back(pcm[j] / 32768.f);
                            return true;
                        },
                        stats);
                } else {
                    // The public upstream API accepts one utterance. This is explicitly
                    // a serial workload, never labelled as native GPU batch inference.
                    for (int lane = 0; lane < batch; ++lane) {
                        breeze::GenRequest req;
                        req.text = requests[lane].text;
                        req.instruction = requests[lane].instruction;
                        req.cfg_scale = cfg;
                        req.seed = 42;
                        req.max_new_tokens = frames;
                        req.split_chars = 0;
                        req.pace = false;
                        req.chunk_first = item.value("chunk_first", 4);
                        req.chunk_max = item.value("chunk_max", 25);
                        req.temperature = .9f;
                        req.top_k = 50;
                        req.top_p = 1.f;
                        req.repetition_penalty = 1.1f;
                        if (cloning) {
                            req.ref_text = transcript;
                            req.ref_codes = reference_codes;
                            req.ref_frames = reference_frames;
                        }
                        breeze::GenTimings tm;
                        breeze::generate(
                            other, codec, req,
                            [&](const float *pcm, int n) {
                                record(lane, n);
                                audio[lane].insert(audio[lane].end(), pcm, pcm + n);
                                return true;
                            },
                            &tm);
                        stages.push_back({{"prompt", tm.prompt},
                                          {"prefill", tm.prefill},
                                          {"backbone", tm.backbone},
                                          {"depth", tm.depth},
                                          {"vocoder", tm.vocoder},
                                          {"frames", tm.frames}});
                    }
                }
                double total = since(start), seconds = 0;
                Json rms = Json::array();
                for (int lane = 0; lane < batch; ++lane) {
                    if (samples[lane] != size_t(frames) * 1920)
                        throw std::runtime_error(
                            "unequal output work: early EOS or missing frames");
                    seconds += samples[lane] / 24000.0;
                    double ss = 0;
                    for (float v : audio[lane]) {
                        if (!std::isfinite(v))
                            throw std::runtime_error("nonfinite audio");
                        ss += double(v) * v;
                    }
                    if (ss / samples[lane] < 1e-12)
                        throw std::runtime_error(
                            "silent output cannot validate a speed comparison");
                    rms.push_back(std::sqrt(ss / samples[lane]));
                }
                Json row = {{"iteration", iteration},
                            {"total_ms", total},
                            {"audio_seconds", seconds},
                            {"audio_seconds_per_second", seconds * 1000 / total},
                            {"first_audio_ms", first},
                            {"time_to_320ms_audio", four_frames},
                            {"completed_ms", completed},
                            {"first_chunk_samples", first_size},
                            {"chunk_counts", chunks},
                            {"rms", rms},
                            {"stages_ms", stages}};
                observations.push_back(row);
                std::cout << Json{{"case", item.at("id")}, {"result", row}}.dump() << std::endl;
            }
            result["cases"].push_back({{"specification", item}, {"observations", observations}});
            std::ofstream(argv[4]) << result.dump(2) << '\n';
        }
        if (!native)
            other.free();
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
