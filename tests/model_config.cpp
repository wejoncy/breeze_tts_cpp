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
    assert(good.codec.transformer.window == a.at("decoder_config").at("sliding_window"));
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
        } catch (const std::exception &) {
            rejected = true;
        }
        assert(rejected);
    }
    for (auto value : {Json(0.5), Json(uint64_t(INT32_MAX) + 1), Json(true)}) {
        auto bad = c;
        bad["codebook_eos_token_id"] = value;
        bool rejected = false;
        try {
            load(bad, a);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        assert(rejected);
    }
    auto small = a;
    small["encoder_config"]["max_position_embeddings"] = 20;
    small["decoder_config"]["max_position_embeddings"] = 32;
    auto limited = load(c, small);
    assert(limited.encoder.transformer.max_positions == 20 && limited.max_reference_frames == 10);
    assert(limited.codec.transformer.max_positions == 32);
    assert(good.depth.rope_low_wave == 8192 && good.depth.rope_high_wave == 2048);
    auto varied = a;
    auto &ec = varied["encoder_config"];
    ec["num_hidden_layers"] = 3;
    ec["hidden_size"] = 384;
    ec["num_attention_heads"] = 6;
    ec["num_key_value_heads"] = 3;
    ec["intermediate_size"] = 768;
    ec["num_filters"] = 48;
    ec["num_residual_layers"] = 2;
    ec["upsampling_ratios"] = {8, 10, 12};
    ec["norm_eps"] = 1e-4;
    ec["rope_theta"] = 25000;
    varied["decoder_config"]["num_hidden_layers"] = 3;
    varied["decoder_config"]["upsample_rates"] = {3, 4, 5, 8};
    auto derived = load(c, varied);
    assert(derived.encoder.ratios == std::vector<int>({12, 10, 8}));
    assert(derived.encoder.transformer.layers == 3 && derived.encoder.transformer.hidden == 384);
    assert(derived.encoder.transformer.kvheads == 3 && derived.encoder.residual_layers == 2);
    assert(derived.encoder.filters == 48 && derived.encoder.transformer.theta == 25000);
    assert(derived.codec.transformer.layers == 3 && derived.codec.upsample_rates.front() == 3);
    auto different_text = c;
    different_text["text_encoder_config"]["num_hidden_layers"] = 2;
    different_text["text_encoder_config"]["layer_types"] = {"full_attention", "sliding_attention"};
    assert(load(different_text, a).text.layers == 2);
    for (const auto &ratios : {Json{8, 0, 5, 4}, Json{8, -1, 5, 4}, Json{8, 6.5, 5, 4},
                               Json{INT32_MAX, INT32_MAX}, Json{3, 7}}) {
        auto bad = a;
        bad["encoder_config"]["upsampling_ratios"] = ratios;
        bool rejected = false;
        try {
            load(c, bad);
        } catch (const std::exception &) {
            rejected = true;
        }
        assert(rejected);
    }
    std::cout << "checkpoint position bounds, token contracts and parsed configs passed\n";
}
