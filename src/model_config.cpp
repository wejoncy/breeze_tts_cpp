#include "model_config.h"
#include "json.h"
#include <algorithm>

namespace tts {
ModelConfig load_model_config(const std::string &dir, int batch, int context, Quantization quantization) {
    if (batch < 1 || batch > 32 || context < 32 || context > 8192)
        throw std::runtime_error("max batch must be 1..32; context 32..8192");
    auto c = read_json(dir + "/config.json");
    auto require = [&](const Json &j, const char *key, const Json &value) {
        if (!j.contains(key) || j[key] != value)
            throw std::runtime_error(std::string("unsupported checkpoint field: ") + key);
    };
    require(c, "model_type", "breeze");
    require(c, "num_codebooks", 16);
    require(c, "audio_num_codebooks", 16);
    require(c, "audio_vocab_size", 2051);
    require(c, "codebook_eos_token_id", 0);
    require(c, "codebook_pad_token_id", 2050);
    require(c, "tie_codebooks_embeddings", true);
    require(c, "text_encoder_proj_type", "linear");
    for (auto spec : std::vector<std::pair<std::string, std::vector<int>>>{
             {"text_encoder_config", {1152, 26, 4, 1, 256, 6912}},
             {"backbone_config", {2048, 28, 16, 8, 128, 6144}},
             {"depth_decoder_config", {1024, 12, 8, 2, 128, 8192}}}) {
        const char *fields[] = {"hidden_size",         "num_hidden_layers", "num_attention_heads",
                                "num_key_value_heads", "head_dim",          "intermediate_size"};
        for (int i = 0; i < 6; i++)
            require(c.at(spec.first), fields[i], spec.second[i]);
    }
    const auto &tc = c.at("text_encoder_config");
    const auto &bc = c.at("backbone_config");
    const auto &dc = c.at("depth_decoder_config");
    require(dc, "model_type", "breeze_depth_decoder_model");
    require(dc, "vocab_size", 2051);
    require(dc, "backbone_hidden_size", bc.at("hidden_size"));
    require(tc, "model_type", "t5gemma2_text");
    require(tc, "hidden_activation", "gelu_pytorch_tanh");
    require(tc, "rms_norm_eps", 1e-6);
    require(tc, "sliding_window", 512);
    require(tc, "query_pre_attn_scalar", 256);
    require(tc, "eoi_token_index", 256000);
    require(tc, "rope_parameters",
            Json{{"full_attention",
                  {{"factor", 8.0}, {"rope_theta", 1000000}, {"rope_type", "linear"}}},
                 {"sliding_attention", {{"rope_theta", 10000}, {"rope_type", "default"}}}});
    Json layer_types = Json::array();
    for (int i = 0; i < 26; ++i)
        layer_types.push_back(i % 6 == 5 ? "full_attention" : "sliding_attention");
    require(tc, "layer_types", layer_types);
    require(bc, "model_type", "qwen3");
    require(bc, "rms_norm_eps", 1e-6);
    require(bc, "rope_theta", 1000000);
    require(bc, "rope_scaling", nullptr);
    require(bc, "use_sliding_window", false);
    require(dc, "rms_norm_eps", 1e-5);
    require(dc, "rope_theta", 500000);
    require(dc, "rope_scaling",
            Json{{"factor", 32.0},
                 {"high_freq_factor", 0.0078125},
                 {"low_freq_factor", 0.001953125},
                 {"original_max_position_embeddings", 16},
                 {"rope_type", "llama3"}});
    for (const auto *cfg : {&tc, &bc, &dc})
        require(*cfg, "attention_bias", false);
    for (const auto *cfg : {&bc, &dc})
        require(*cfg, "hidden_act", "silu");
    for (const auto *cfg : {&bc, &dc})
        if (cfg->contains("mlp_bias")) // Qwen3 omits this field; its MLP is bias-free.
            require(*cfg, "mlp_bias", false);
    auto a = read_json(dir + "/audio_tokenizer/config.json");
    require(a, "model_type", "qwen3_tts_tokenizer_12hz");
    require(a, "decode_upsample_rate", 1920);
    require(a, "output_sample_rate", 24000);
    require(a, "encoder_valid_num_quantizers", 16);
    require(a, "input_sample_rate", 24000);
    require(a, "encode_downsample_rate", 1920);
    const auto &ec = a.at("encoder_config");
    const Json encoder_expected = {{"audio_channels", 1},
                                   {"hidden_size", 512},
                                   {"num_hidden_layers", 8},
                                   {"num_attention_heads", 8},
                                   {"num_key_value_heads", 8},
                                   {"head_dim", 64},
                                   {"intermediate_size", 2048},
                                   {"hidden_act", "gelu"},
                                   {"norm_eps", 1e-5},
                                   {"rope_theta", 10000},
                                   {"attention_bias", false},
                                   {"num_filters", 64},
                                   {"num_residual_layers", 1},
                                   {"kernel_size", 7},
                                   {"last_kernel_size", 3},
                                   {"residual_kernel_size", 3},
                                   {"compress", 2},
                                   {"upsampling_ratios", {8, 6, 5, 4}},
                                   {"codebook_size", 2048},
                                   {"codebook_dim", 256},
                                   {"num_semantic_quantizers", 1},
                                   {"normalize", false},
                                   {"pad_mode", "constant"},
                                   {"use_causal_conv", true},
                                   {"use_conv_shortcut", false}};
    for (const auto &item : encoder_expected.items())
        require(ec, item.key().c_str(), item.value());
    const auto &ac = a.at("decoder_config");
    const Json expected = {{"latent_dim", 1024},
                           {"codebook_dim", 512},
                           {"codebook_size", 2048},
                           {"decoder_dim", 1536},
                           {"hidden_size", 512},
                           {"intermediate_size", 1024},
                           {"head_dim", 64},
                           {"num_attention_heads", 16},
                           {"num_key_value_heads", 16},
                           {"num_hidden_layers", 8},
                           {"num_quantizers", 16},
                           {"num_semantic_quantizers", 1},
                           {"rms_norm_eps", 1e-5},
                           {"rope_theta", 10000},
                           {"sliding_window", 72},
                           {"upsample_rates", {8, 5, 4, 3}},
                           {"upsampling_ratios", {2, 2}},
                           {"attention_bias", false},
                           {"hidden_act", "silu"}};
    for (const auto &item : expected.items())
        require(ac, item.key().c_str(), item.value());
    auto transformer = [](const Json &j, int positions) {
        TransformerConfig result{};
        result.hidden = j.at("hidden_size");
        result.layers = j.at("num_hidden_layers");
        result.heads = j.at("num_attention_heads");
        result.kvheads = j.at("num_key_value_heads");
        result.dim = j.at("head_dim");
        result.intermediate = j.at("intermediate_size");
        result.eps = j.at("rms_norm_eps");
        result.theta = j.value("rope_theta", 0.f);
        int model_limit = j.at("max_position_embeddings");
        if (positions <= 0 || positions > model_limit)
            throw std::invalid_argument("requested positions exceed checkpoint configuration");
        result.max_positions = positions;
        return result;
    };
    ModelConfig result;
    result.codebooks = c.at("num_codebooks");
    result.sample_rate = a.at("output_sample_rate");
    result.samples_per_frame = a.at("decode_upsample_rate");
    int encoder_limit = ec.at("max_position_embeddings");
    // The validated convolution stack emits two transformer positions per audio frame.
    result.max_reference_frames = std::min(context - 2, encoder_limit / 2);
    if (result.max_reference_frames <= 0)
        throw std::invalid_argument("invalid encoder position limit");
    result.encoder_positions = result.max_reference_frames * 2;
    result.text = transformer(tc, context);
    result.text.gemma = result.text.qknorm = true;
    result.text.causal = false;
    result.text.window = tc.at("sliding_window");
    const auto &text_rope = tc.at("rope_parameters");
    result.text.theta = text_rope.at("sliding_attention").at("rope_theta");
    result.text.full_theta = text_rope.at("full_attention").at("rope_theta");
    result.text.full_factor = text_rope.at("full_attention").at("factor");
    for (const auto &kind : tc.at("layer_types"))
        result.text.full_attention.push_back(kind == "full_attention");
    result.backbone = transformer(bc, context);
    result.backbone.qknorm = true;
    // Initial context + first code consume two positions, followed by codebooks-2.
    result.depth = transformer(dc, result.codebooks);
    result.depth.rope_type = 2;
    const auto &scaling = dc.at("rope_scaling");
    result.depth.factor = scaling.at("factor");
    float original = scaling.at("original_max_position_embeddings");
    result.depth.rope_low_wave = original / scaling.at("low_freq_factor").get<float>();
    result.depth.rope_high_wave = original / scaling.at("high_freq_factor").get<float>();
    result.codec = transformer(ac, ac.at("max_position_embeddings"));
    result.codec.type = Type::F32;
    result.codec.window = ac.at("sliding_window");
    result.text.quantization = result.backbone.quantization = result.depth.quantization = quantization;
    return result;
}
} // namespace tts
