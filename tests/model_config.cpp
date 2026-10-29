#include "model_config.h"
#include "json.h"
#include <cassert>
#include <filesystem>
#include <iostream>
#include <chrono>

using namespace tts;
int main(int argc, char **argv) {
    const std::string source = argc > 1 ? argv[1] : "..";
    auto c = read_json(source + "/config.json");
    auto a = read_json(source + "/audio_tokenizer/config.json");
    auto directory = std::filesystem::temp_directory_path() /
                     ("tts-配置-" +
                      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(directory / "audio_tokenizer");
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() {
            std::filesystem::remove_all(path);
        }
    } cleanup{directory};
    auto load = [&](const Json &model, const Json &audio) {
        std::ofstream(directory / "config.json") << model;
        std::ofstream(directory / "audio_tokenizer/config.json") << audio;
        return load_model_config(directory.u8string(), 4, 256);
    };
    auto good = load(c, a);
    assert(good.backbone.max_positions == 256 && good.depth.max_positions == 16);
    assert(good.codec.window == a.at("decoder_config").at("sliding_window"));
    for (const char *component :
         {"text_encoder_config", "backbone_config", "depth_decoder_config"}) {
        auto bad = c;
        bad[component]["max_position_embeddings"] = 15;
        bool rejected = false;
        try {
            load(bad, a);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        assert(rejected);
    }
    for (const char *field : {"codebook_eos_token_id", "audio_vocab_size"}) {
        auto bad = c;
        bad[field] = -1;
        bool rejected = false;
        try {
            load(bad, a);
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        assert(rejected);
    }
    auto small = a;
    small["encoder_config"]["max_position_embeddings"] = 20;
    small["decoder_config"]["max_position_embeddings"] = 32;
    auto limited = load(c, small);
    assert(limited.encoder_positions == 20 && limited.max_reference_frames == 10);
    assert(limited.codec.max_positions == 32);
    assert(good.depth.rope_low_wave == 8192 && good.depth.rope_high_wave == 2048);
    std::cout << "checkpoint position bounds, token contracts and parsed configs passed\n";
}
