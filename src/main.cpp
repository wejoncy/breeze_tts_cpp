#include "engine.h"
#include "log.h"
#include "json.h"
#include "tokenizer.h"
#include <map>
#include "audio_io.h"
#include "wav.h"
#include <iostream>
#include <filesystem>

int run_cli(int argc, const char *const *argv) {
    try {
        std::string model = "..", text, instruction, output = "speech.wav", batchfile, streamfile,
                    codefile, encodefile, ref_audio, ref_text, speaker = "S0", quantization;
        tts_log_config logging{TTS_LOG_INFO, nullptr, 0, nullptr, nullptr};
        std::string log_file;
        bool cfg_ref_set = false, cfg_ins_set = false;
        int device = 0, context = 2048, repeat = 1;
        bool tokenize = false;
        tts::GenerationOptions opts;
        for (int i = 1; i < argc; i++) {
            std::string a = argv[i];
            auto value = [&]() {
                if (i + 1 >= argc)
                    throw std::runtime_error("missing value for " + a);
                return std::string(argv[++i]);
            };
            if (a == "--model")
                model = value();
            else if (a == "--text")
                text = value();
            else if (a == "--instruction")
                instruction = value();
            else if (a == "--ref-audio")
                ref_audio = value();
            else if (a == "--ref-text")
                ref_text = value();
            else if (a == "--speaker")
                speaker = value();
            else if (a == "--top-p")
                opts.top_p = opts.depth_top_p = std::stof(value());
            else if (a == "--repetition-penalty")
                opts.repetition_penalty = std::stof(value());
            else if (a == "--no-sample")
                opts.do_sample = false;
            else if (a == "--depth-no-sample")
                opts.depth_do_sample = false;
            else if (a == "--depth-temperature")
                opts.depth_temperature = std::stof(value());
            else if (a == "--depth-top-k")
                opts.depth_top_k = std::stoi(value());
            else if (a == "--depth-top-p")
                opts.depth_top_p = std::stof(value());
            else if (a == "--cfg-scale-ref") {
                opts.cfg_scale_ref = std::stof(value());
                cfg_ref_set = true;
            } else if (a == "--cfg-scale-ins") {
                opts.cfg_scale_ins = std::stof(value());
                cfg_ins_set = true;
            } else if (a == "--output")
                output = value();
            else if (a == "--batch")
                batchfile = value();
            else if (a == "--stream")
                streamfile = value();
            else if (a == "--max-frames" || a == "--max-new-tokens")
                opts.max_frames = std::stoi(value());
            else if (a == "--temperature")
                opts.depth_temperature = opts.temperature = std::stof(value());
            else if (a == "--top-k")
                opts.depth_top_k = opts.top_k = std::stoi(value());
            else if (a == "--cfg-scale")
                opts.cfg_scale = std::stof(value());
            else if (a == "--seed")
                opts.seed = std::stoull(value());
            else if (a == "--device")
                device = std::stoi(value());
            else if (a == "--context")
                context = std::stoi(value());
            else if (a == "--quantization")
                quantization = value();
            else if (a == "--log-level")
                logging.level = tts::log::parse_level(value());
            else if (a == "--log-file")
                log_file = value();
            else if (a == "--log-json")
                logging.json = 1;
            else if (a == "--repeat")
                repeat = std::stoi(value());
            else if (a == "--no-graph")
                opts.use_cuda_graph = 0;
            else if (a == "--tokenize")
                tokenize = true;
            else if (a == "--encode-audio")
                encodefile = value();
            else if (a == "--decode-codes")
                codefile = value();
            else if (a == "--help") {
                std::cout
                    << "tts --model DIR --text TEXT [--instruction TEXT] [--cfg-scale 4]\n"
                       "    [--output speech.wav] [--stream audio.pcm|-] [--batch requests.jsonl]\n"
                       "    [--max-frames 750] [--temperature .9|0] [--top-k 50] [--seed 42]\n"
                       "    [--ref-audio FILE --ref-text TEXT] [--speaker S0]\n"
                       "    [--cfg-scale-ref 2 --cfg-scale-ins 3] [--top-p .95] "
                       "[--repetition-penalty 1.1]\n"
                       "    [--depth-temperature .9] [--depth-top-k 50] [--depth-top-p 1]\n"
                       "    [--no-sample] [--depth-no-sample] [--encode-audio FILE --output "
                       "codes.i32]\n"
                       "    [--repeat 3] [--no-graph] [--context 2048] [--device 0] "
                       "[--quantization "
                       "fp8|int8]\n"
                       "    [--log-level trace|debug|info|warn|error|off] [--log-file FILE] "
                       "[--log-json]\n"
                       "    [--tokenize] [--decode-codes frames.i32 --output waveform.f32]\n"
                       "Batch JSONL: {\"text\":\"...\",\"instruction\":\"...\"}. Outputs: "
                       "PREFIX.0.wav etc.\n";
                return 0;
            } else
                throw std::runtime_error("unknown argument: " + a);
        }
        logging.file_path = log_file.empty() ? nullptr : log_file.c_str();
        tts::log::configure(logging);
        if (tokenize) {
            tts::Tokenizer tok(model + "/tokenizer.json");
            std::cout << tts::Json(tok.encode(text)).dump() << "\n";
            return 0;
        }
        if (repeat < 1 || repeat > 1000)
            throw std::runtime_error("repeat must be 1..1000");
        if (cfg_ref_set != cfg_ins_set)
            throw std::invalid_argument("both dual CFG scales are required");
        opts.dual_cfg = cfg_ref_set;
        std::vector<tts::Json> inputs;
        if (!batchfile.empty()) {
            std::ifstream f(std::filesystem::u8path(batchfile));
            if (!f)
                throw std::runtime_error("cannot open batch file");
            std::string line;
            while (std::getline(f, line)) {
                if (line.empty())
                    continue;
                auto j = tts::Json::parse(line);
                inputs.push_back(std::move(j));
            }
        } else
            inputs.push_back(tts::Json{{"text", text}, {"instruction", instruction}});
        if (inputs.empty() || inputs.size() > 32)
            throw std::runtime_error("batch size must be 1..32");
        if (inputs.size() > 1 && !streamfile.empty())
            throw std::runtime_error(
                "CLI raw stream supports one request; use ABI for batch streaming");
        tts::Engine engine(model, device, inputs.size(), context, 128,
                           tts::quantization_from_string(quantization));
        const auto format = engine.audio_format();
        if (!encodefile.empty()) {
            auto codes = engine.encode_audio(tts::read_audio_file(
                encodefile, engine.max_reference_samples(), format.sample_rate));
            std::ofstream out(std::filesystem::u8path(output), std::ios::binary);
            out.write(reinterpret_cast<const char *>(codes.data()), codes.size() * 4);
            if (!out)
                throw std::runtime_error("cannot write reference codes");
            return 0;
        }
        if (!codefile.empty()) {
            std::ifstream f(std::filesystem::u8path(codefile), std::ios::binary | std::ios::ate);
            if (!f)
                throw std::runtime_error("cannot open codes");
            auto size = f.tellg();
            if (size < 0 || size % (format.codebooks * sizeof(int32_t)))
                throw std::runtime_error("invalid codes length");
            std::vector<int> codes(size / 4);
            f.seekg(0);
            f.read(reinterpret_cast<char *>(codes.data()), size);
            auto pcm = engine.decode_codes(codes);
            std::ofstream out(std::filesystem::u8path(output), std::ios::binary);
            out.write(reinterpret_cast<char *>(pcm.data()), pcm.size() * 4);
            if (!out)
                throw std::runtime_error("cannot write decoded waveform");
            return 0;
        }
        std::vector<tts::Request> requests;
        std::map<std::pair<std::string, std::string>, std::shared_ptr<const tts::Reference>>
            references;
        for (auto &i : inputs) {
            tts::Request request{i.at("text").get<std::string>(),
                                 i.value("instruction", instruction), i.value("speaker", speaker),
                                 nullptr};
            auto audio = i.value("ref_audio_path", i.value("ref_audio", ref_audio));
            auto transcript = i.value("ref_text", ref_text);
            if (audio.empty() != transcript.empty())
                throw std::invalid_argument("ref-audio and ref-text must be provided together");
            if (!audio.empty()) {
                auto key = std::make_pair(audio, transcript);
                if (!references.count(key))
                    references[key] = std::make_shared<const tts::Reference>(tts::Reference{
                        transcript,
                        engine.encode_audio(tts::read_audio_file(
                            audio, engine.max_reference_samples(), format.sample_rate)),
                        format.codebooks, format.codebook_size});
                request.reference = references.at(key);
            }
            requests.push_back(std::move(request));
        }
        for (int iteration = 0; iteration < repeat; iteration++) {
            std::vector<std::vector<int16_t>> audio(inputs.size());
            std::ofstream stream;
            if (!streamfile.empty() && streamfile != "-") {
                stream.open(std::filesystem::u8path(streamfile), std::ios::binary);
                if (!stream)
                    throw std::runtime_error("cannot open PCM output");
            }
            tts::Stats stats{};
            engine.generate(
                requests, opts,
                [&](int b, const int16_t *pcm, size_t n) {
                    audio[b].insert(audio[b].end(), pcm, pcm + n);
                    if (!streamfile.empty()) {
                        auto &out = streamfile == "-" ? std::cout : stream;
                        out.write(reinterpret_cast<const char *>(pcm), n * 2);
                        out.flush();
                        if (!out)
                            return false;
                    }
                    return true;
                },
                stats);
            for (size_t b = 0; b < audio.size(); b++)
                tts::write_wav(audio.size() == 1 ? output
                                                 : output + "." + std::to_string(b) + ".wav",
                               audio[b], format.sample_rate);
        }
    } catch (const std::exception &e) {
        TTS_LOG(TTS_LOG_ERROR, "cli", "{}", e.what());
        return 1;
    }
    return 0;
}
