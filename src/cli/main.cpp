#include "tts/engine.h"
#include "support/log.h"
#include "io/json.h"
#include "io/tokenizer.h"
#include <map>
#include "io/audio_io.h"
#include "io/wav.h"
#include <iostream>
#include <filesystem>
#include <limits>
#include <variant>
#include <type_traits>

int run_cli(int argc, const char *const *argv) {
    try {
        std::string model = "..", text, instruction, output = "speech.wav", batchfile, streamfile,
                    codefile, encodefile, ref_audio, ref_text, speaker = "S0", quantization,
                    language, prompt_language, text_split_method;
        double fragment_interval = 0;
        tts_log_config logging{TTS_LOG_INFO, nullptr, 0};
        std::string log_file;
        bool cfg_ref_set = false, cfg_ins_set = false;
        int device = 0, context = 2048, repeat = 1;
        size_t runtime_memory_mb = 0;
        size_t memory_limit_mb = 0;
        bool tokenize = false;
        using Options = tts::GenerationOptions;
        using Override = std::variant<std::pair<int Options::*, int>,
            std::pair<float Options::*, float>, std::pair<bool Options::*, bool>,
            std::pair<uint64_t Options::*, uint64_t>, std::pair<double Options::*, double>,
            std::pair<tts::SynthesisMode Options::*, tts::SynthesisMode>>;
        std::vector<Override> overrides;
        auto set = [&](auto field, auto v) {
            using Value = std::decay_t<decltype(std::declval<Options>().*field)>;
            overrides.emplace_back(std::pair<Value Options::*, Value>{field, static_cast<Value>(v)});
        };
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
            else if (a == "--language")
                language = value();
            else if (a == "--prompt-language")
                prompt_language = value();
            else if (a == "--text-split-method")
                text_split_method = value();
            else if (a == "--fragment-interval")
                fragment_interval = std::stod(value());
            else if (a == "--top-p")
                {
                auto v = std::stof(value());
                set(&Options::top_p, v);
                set(&Options::depth_top_p, v);
            }
            else if (a == "--repetition-penalty")
                set(&Options::repetition_penalty, std::stof(value()));
            else if (a == "--no-sample")
                set(&Options::do_sample, false);
            else if (a == "--depth-no-sample")
                set(&Options::depth_do_sample, false);
            else if (a == "--depth-temperature")
                set(&Options::depth_temperature, std::stof(value()));
            else if (a == "--depth-top-k")
                set(&Options::depth_top_k, std::stoi(value()));
            else if (a == "--depth-top-p")
                set(&Options::depth_top_p, std::stof(value()));
            else if (a == "--cfg-scale-ref") {
                set(&Options::cfg_scale_ref, std::stof(value()));
                cfg_ref_set = true;
            } else if (a == "--cfg-scale-ins") {
                set(&Options::cfg_scale_ins, std::stof(value()));
                cfg_ins_set = true;
            } else if (a == "--output")
                output = value();
            else if (a == "--batch")
                batchfile = value();
            else if (a == "--stream")
                streamfile = value();
            else if (a == "--max-frames" || a == "--max-new-tokens")
                set(&Options::max_frames, std::stoi(value()));
            else if (a == "--temperature")
                {
                auto v = std::stof(value());
                set(&Options::depth_temperature, v);
                set(&Options::temperature, v);
            }
            else if (a == "--top-k")
                {
                auto v = std::stoi(value());
                set(&Options::depth_top_k, v);
                set(&Options::top_k, v);
            }
            else if (a == "--cfg-scale")
                set(&Options::cfg_scale, std::stof(value()));
            else if (a == "--speed-factor")
                set(&Options::speed_factor, std::stod(value()));
            else if (a == "--synthesis-mode")
                set(&Options::synthesis_mode, tts::parse_synthesis_mode(value()));
            else if (a == "--streaming-chunk-seconds")
                set(&Options::streaming_chunk_seconds, std::stod(value()));
            else if (a == "--streaming-lookahead-seconds")
                set(&Options::streaming_lookahead_seconds, std::stod(value()));
            else if (a == "--streaming-overlap-seconds")
                set(&Options::streaming_overlap_seconds, std::stod(value()));
            else if (a == "--semantic-lookahead-seconds")
                set(&Options::semantic_lookahead_seconds, std::stod(value()));
            else if (a == "--sample-steps")
                set(&Options::sample_steps, std::stoi(value()));
            else if (a == "--cfg-rate")
                set(&Options::cfg_rate, std::stof(value()));
            else if (a == "--seed")
                set(&Options::seed, std::stoull(value()));
            else if (a == "--device")
                device = std::stoi(value());
            else if (a == "--runtime-memory-mb" || a == "--memory-limit-mb") {
                auto amount = std::stoull(value());
                if (amount > SIZE_MAX / (1 << 20))
                    throw std::invalid_argument("runtime memory budget is too large");
                (a == "--memory-limit-mb" ? memory_limit_mb : runtime_memory_mb) = size_t(amount);
            } else if (a == "--context")
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
                set(&Options::use_cuda_graph, false);
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
                       "    [--max-frames N] [--temperature VALUE] [--top-k N] [--seed N]\n"
                       "    [--ref-audio FILE --ref-text TEXT] [--speaker S0]\n"
                       "    [--language LANG] [--prompt-language LANG]\n"
                       "    [--text-split-method cut0..cut5] [--fragment-interval SECONDS]\n"
                       "    [--speed-factor 1] [--sample-steps 4] [--cfg-rate 0]\n"
                       "    [--synthesis-mode offline|fixed|hybrid] [--streaming-chunk-seconds SECONDS]\n"
                       "    [--streaming-lookahead-seconds SECONDS] [--streaming-overlap-seconds SECONDS]\n"
                       "    [--semantic-lookahead-seconds SECONDS]\n"
                       "    [--cfg-scale-ref 2 --cfg-scale-ins 3] [--top-p .95] "
                       "[--repetition-penalty 1.1]\n"
                       "    [--depth-temperature .9] [--depth-top-k 50] [--depth-top-p 1]\n"
                       "    [--no-sample] [--depth-no-sample] [--encode-audio FILE --output "
                       "codes.i32]\n"
                       "    [--repeat 3] [--no-graph] [--context 2048] [--device 0] "
                       "[--quantization "
                       "int8|fp8|group_int8|group_fp8] [--memory-limit-mb 0] [--runtime-memory-mb 0]\n"
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
        set(&Options::dual_cfg, cfg_ref_set);
        std::vector<tts::Json> inputs;
        if (!batchfile.empty()) {
            std::ifstream f(std::filesystem::u8path(batchfile));
            if (!f)
                throw std::runtime_error("cannot open batch file");
            std::string line;
            while (std::getline(f, line)) {
                if (line.empty())
                    continue;
                auto j = tts::parse_json(line);
                inputs.push_back(std::move(j));
            }
        } else
            inputs.push_back(tts::Json{{"text", text}, {"instruction", instruction}});
        if (inputs.empty() || inputs.size() > size_t(std::numeric_limits<int>::max()))
            throw std::runtime_error("batch size must fit a positive engine dimension");
        if (inputs.size() > 1 && !streamfile.empty())
            throw std::runtime_error(
                "CLI raw stream supports one request; use ABI for batch streaming");
        tts::Engine engine(model, device, int(inputs.size()), context, 128,
                           tts::quantization_from_string(quantization),
                           runtime_memory_mb * (1 << 20), memory_limit_mb * (1 << 20));
        auto opts = engine.default_options();
        for (const auto &value : overrides)
            std::visit([&](const auto &field) { opts.*field.first = field.second; }, value);
        engine.run_async();
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
            if (format.codebooks <= 0)
                throw std::runtime_error("loaded model does not expose discrete audio codes");
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
        std::vector<tts::Input> requests;
        std::map<std::tuple<std::string, std::string, std::string>, std::shared_ptr<const tts::Reference>>
            references;
        for (auto &i : inputs) {
            tts::Input request{i.at("text").get<std::string>(), i.value("instruction", instruction),
                               i.value("speaker", speaker), nullptr, i.value("language", language),
                               i.value("text_split_method", text_split_method),
                               i.value("fragment_interval", fragment_interval)};
            auto audio = i.value("ref_audio_path", i.value("ref_audio", ref_audio));
            auto transcript = i.value("ref_text", ref_text);
            auto reference_language = i.value("prompt_language", prompt_language);
            if (audio.empty() != transcript.empty())
                throw std::invalid_argument("ref-audio and ref-text must be provided together");
            if (!audio.empty()) {
                auto key = std::make_tuple(audio, transcript, reference_language);
                if (!references.count(key))
                    references[key] = engine.prepare_reference(tts::read_audio_file(
                        audio, engine.max_reference_samples(), format.sample_rate), transcript,
                        reference_language);
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
