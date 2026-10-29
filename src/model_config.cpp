#include "model_config.h"
#include "json.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace tts {
namespace {
int integer(const Json &j, const char *key, int minimum) {
    const auto &v = j.at(key);
    if (!v.is_number_integer() || v < minimum || v > INT32_MAX)
        throw std::invalid_argument(std::string("integer outside supported range: ") + key);
    return v.get<int>();
}
int positive(const Json &j, const char *key) {
    return integer(j, key, 1);
}
float positive_float(const Json &j, const char *key) {
    float v = j.at(key).get<float>();
    if (!std::isfinite(v) || v <= 0)
        throw std::invalid_argument(std::string("expected positive finite value: ") + key);
    return v;
}
std::vector<int> positive_list(const Json &j, const char *key) {
    const auto &values = j.at(key);
    if (!values.is_array())
        throw std::invalid_argument(std::string("expected array: ") + key);
    std::vector<int> result;
    for (const auto &v : values) {
        if (!v.is_number_integer() || v.get<int64_t>() <= 0 || v.get<int64_t>() > INT32_MAX)
            throw std::invalid_argument(std::string("expected positive integer in ") + key);
        result.push_back(v.get<int>());
    }
    return result;
}
int product(const std::vector<int> &values) {
    int64_t result = 1;
    for (int v : values) {
        if (v <= 0 || result > INT32_MAX / int64_t(v))
            throw std::invalid_argument("invalid or overflowing model dimensions/strides");
        result *= v;
    }
    return int(result);
}
} // namespace
ModelConfig load_model_config(const std::string &dir, int batch, int context, Quantization quantization) {
    if (batch < 1 || batch > 32 || context < 32 || context > 8192)
        throw std::runtime_error("max batch must be 1..32; context 32..8192");
    auto c = read_json(dir + "/config.json");
    auto require = [&](const Json &j, const char *key, const Json &value) {
        if (!j.contains(key) || j[key] != value)
            throw std::runtime_error(std::string("unsupported checkpoint field: ") + key);
    };
    require(c, "model_type", "breeze");
    require(c, "tie_codebooks_embeddings", true);
    require(c, "text_encoder_proj_type", "linear");
    const auto &tc = c.at("text_encoder_config");
    const auto &bc = c.at("backbone_config");
    const auto &dc = c.at("depth_decoder_config");
    require(dc, "model_type", "breeze_depth_decoder_model");
    require(dc, "backbone_hidden_size", bc.at("hidden_size"));
    require(tc, "model_type", "t5gemma2_text");
    require(tc, "hidden_activation", "gelu_pytorch_tanh");
    require(tc.at("rope_parameters").at("full_attention"), "rope_type", "linear");
    require(tc.at("rope_parameters").at("sliding_attention"), "rope_type", "default");
    require(bc, "model_type", "qwen3");
    require(bc, "rope_scaling", nullptr);
    require(bc, "use_sliding_window", false);
    require(dc.at("rope_scaling"), "rope_type", "llama3");
    for (const auto *cfg : {&tc, &bc, &dc})
        require(*cfg, "attention_bias", false);
    for (const auto *cfg : {&bc, &dc})
        require(*cfg, "hidden_act", "silu");
    for (const auto *cfg : {&bc, &dc})
        if (cfg->contains("mlp_bias")) // Qwen3 omits this field; its MLP is bias-free.
            require(*cfg, "mlp_bias", false);
    auto a = read_json(dir + "/audio_tokenizer/config.json");
    require(a, "model_type", "qwen3_tts_tokenizer_12hz");
    const auto &ec = a.at("encoder_config"), &ac = a.at("decoder_config");
    require(ec, "audio_channels", 1); // Input adapter supplies mono audio.
    require(ec, "hidden_act", "gelu");
    require(ec, "attention_bias", false);
    require(ec, "normalize", false);
    require(ec, "pad_mode", "constant");
    require(ec, "use_causal_conv", true);
    require(ec, "use_conv_shortcut", false);
    require(ac, "attention_bias", false);
    require(ac, "hidden_act", "silu");
    auto transformer = [](const Json &j, int positions, const char *eps = "rms_norm_eps") {
        TransformerConfig result{};
        result.hidden = positive(j, "hidden_size");
        result.layers = positive(j, "num_hidden_layers");
        result.heads = positive(j, "num_attention_heads");
        result.kvheads = positive(j, "num_key_value_heads");
        result.dim = positive(j, "head_dim");
        result.intermediate = positive(j, "intermediate_size");
        product({2, result.intermediate});
        result.eps = positive_float(j, eps);
        result.theta = j.contains("rope_theta") ? positive_float(j, "rope_theta") : 0;
        if (result.heads % result.kvheads || result.dim % 2)
            throw std::invalid_argument(
                "attention requires divisible Q/KV heads and an even head dimension");
        if (int64_t(result.heads) + 2LL * result.kvheads > INT32_MAX)
            throw std::invalid_argument("attention head count overflow");
        product({result.heads + 2 * result.kvheads, result.dim});
        int model_limit = positive(j, "max_position_embeddings");
        if (positions <= 0 || positions > model_limit)
            throw std::invalid_argument("requested positions exceed checkpoint configuration");
        result.max_positions = positions;
        return result;
    };
    ModelConfig result{};
    result.codebooks = positive(c, "num_codebooks");
    result.audio_vocab = positive(c, "audio_vocab_size");
    result.backbone_eos =
        positive(c, "vocab_size"); // Breeze appends EOS after the backbone vocabulary.
    result.codebook_eos = integer(c, "codebook_eos_token_id", 0);
    result.text_eoi = integer(tc, "eoi_token_index", 0);
    result.text_vocab = positive(c, "text_vocab_size");
    require(tc, "vocab_size", result.text_vocab);
    const auto &instruction = c.at("text_encoder_special_tokens_config").at("instruction_tokens");
    result.instruction_bos = instruction.at("bos").get<std::string>();
    result.instruction_eos = instruction.at("eos").get<std::string>();
    if (result.instruction_bos.empty() || result.instruction_eos.empty())
        throw std::invalid_argument("empty instruction token delimiters");
    require(c, "audio_num_codebooks", result.codebooks);
    require(dc, "vocab_size", result.audio_vocab);
    if (result.codebooks < 2 || result.codebook_eos < 0 ||
        result.codebook_eos >= result.audio_vocab || result.text_eoi < 0 ||
        result.text_eoi >= positive(tc, "vocab_size"))
        throw std::invalid_argument("invalid codebook or text token configuration");
    result.sample_rate = positive(a, "output_sample_rate");
    if (result.sample_rate > 768000)
        throw std::invalid_argument("sample rate exceeds the audio adapter limit");
    result.samples_per_frame = positive(a, "decode_upsample_rate");
    require(a, "input_sample_rate", result.sample_rate);
    require(a, "encode_downsample_rate", result.samples_per_frame);
    require(ec, "sampling_rate", result.sample_rate);
    require(a, "encoder_valid_num_quantizers", result.codebooks);
    require(ac, "num_quantizers", result.codebooks);
    auto &encoder = result.encoder;
    encoder.ratios = positive_list(ec, "upsampling_ratios");
    std::reverse(encoder.ratios.begin(), encoder.ratios.end());
    encoder.stride = product(encoder.ratios);
    if (result.samples_per_frame % encoder.stride)
        throw std::invalid_argument("encoder strides do not divide the configured frame size");
    encoder.downsample_stride = result.samples_per_frame / encoder.stride;
    encoder.filters = positive(ec, "num_filters");
    encoder.residual_layers = positive(ec, "num_residual_layers");
    encoder.residual_kernel = positive(ec, "residual_kernel_size");
    encoder.dilation_growth = positive(ec, "dilation_growth_rate");
    encoder.compress = positive(ec, "compress");
    int channels = encoder.filters, dilation = 1;
    for (size_t i = 0; i < encoder.ratios.size(); ++i) {
        channels = product({channels, 2});
        product({encoder.ratios[i], 2});
    }
    if (encoder.residual_layers > INT32_MAX - 2 || encoder.ratios.size() > size_t(INT32_MAX - 1))
        throw std::invalid_argument("encoder layer indexing overflow");
    product({int(encoder.ratios.size()) + 1, encoder.residual_layers + 2});
    for (int i = 1; i < encoder.residual_layers; ++i)
        dilation = product({dilation, encoder.dilation_growth});
    product({dilation, encoder.residual_kernel});
    encoder.first_kernel = positive(ec, "kernel_size");
    encoder.last_kernel = positive(ec, "last_kernel_size");
    encoder.codebooks = result.codebooks;
    encoder.semantic_codebooks = positive(ec, "num_semantic_quantizers");
    encoder.codebook_size = positive(ec, "codebook_size");
    encoder.codebook_dim = positive(ec, "codebook_dim");
    require(ec, "vector_quantization_hidden_dimension", encoder.codebook_dim);
    if (encoder.semantic_codebooks >= encoder.codebooks ||
        encoder.codebooks > positive(ec, "num_quantizers") ||
        encoder.filters / encoder.compress < 1)
        throw std::invalid_argument("invalid encoder quantizer/channel configuration");
    int encoder_limit = positive(ec, "max_position_embeddings");
    result.max_reference_frames = std::min(context - 2, encoder_limit / encoder.downsample_stride);
    if (result.max_reference_frames <= 0)
        throw std::invalid_argument("invalid encoder position limit");
    encoder.transformer =
        transformer(ec, result.max_reference_frames * encoder.downsample_stride, "norm_eps");
    encoder.transformer.type = Type::F32;
    if (encoder.transformer.hidden != product({encoder.transformer.heads, encoder.transformer.dim}))
        throw std::invalid_argument("encoder hidden size must equal its query width");
    auto &codec = result.codec;
    codec.transformer = transformer(ac, positive(ac, "max_position_embeddings"));
    codec.transformer.type = Type::F32;
    codec.transformer.window = positive(ac, "sliding_window");
    codec.latent_dim = positive(ac, "latent_dim");
    codec.codebook_dim = positive(ac, "codebook_dim");
    codec.decoder_dim = positive(ac, "decoder_dim");
    codec.upsampling_ratios = positive_list(ac, "upsampling_ratios");
    codec.upsample_rates = positive_list(ac, "upsample_rates");
    codec.samples_per_frame =
        product({product(codec.upsampling_ratios), product(codec.upsample_rates)});
    int decoder_channels = codec.decoder_dim;
    for (int rate : codec.upsample_rates) {
        product({rate, 2});
        if (decoder_channels < 2)
            throw std::invalid_argument("too many codec upsampling stages for decoder_dim");
        decoder_channels /= 2;
    }
    codec.codebooks = result.codebooks;
    codec.semantic_codebooks = positive(ac, "num_semantic_quantizers");
    codec.codebook_size = positive(ac, "codebook_size");
    if (codec.samples_per_frame != result.samples_per_frame ||
        codec.codebook_size != encoder.codebook_size ||
        codec.semantic_codebooks >= codec.codebooks || codec.codebook_dim % 2)
        throw std::invalid_argument("inconsistent audio codec configuration");
    if (codec.codebook_size >= result.audio_vocab || result.backbone_eos < codec.codebook_size)
        throw std::invalid_argument("audio vocabulary does not cover codec codes and EOS");
    // Sampling uses one block sort with 4096 slots; reject unsupported vocabularies at load.
    if (result.audio_vocab > 4096 || result.backbone_eos >= 4096)
        throw std::invalid_argument("vocabulary exceeds the sampling kernel capacity");
    product({result.codebooks, result.audio_vocab});
    result.text = transformer(tc, context);
    result.text.gemma = result.text.qknorm = true;
    result.text.causal = false;
    result.text.window = positive(tc, "sliding_window");
    const auto &text_rope = tc.at("rope_parameters");
    result.text.theta = positive_float(text_rope.at("sliding_attention"), "rope_theta");
    result.text.full_theta = positive_float(text_rope.at("full_attention"), "rope_theta");
    result.text.full_factor = positive_float(text_rope.at("full_attention"), "factor");
    if (tc.at("layer_types").size() != size_t(result.text.layers))
        throw std::invalid_argument("text layer_types must match num_hidden_layers");
    for (const auto &kind : tc.at("layer_types")) {
        if (kind != "full_attention" && kind != "sliding_attention")
            throw std::invalid_argument("unsupported text attention kind");
        result.text.full_attention.push_back(kind == "full_attention");
    }
    require(tc, "query_pre_attn_scalar", result.text.dim);
    result.backbone = transformer(bc, context);
    result.backbone.qknorm = true;
    // Initial context + first code consume two positions, followed by codebooks-2.
    result.depth = transformer(dc, result.codebooks);
    result.depth.rope_type = 2;
    const auto &scaling = dc.at("rope_scaling");
    result.depth.factor = positive_float(scaling, "factor");
    float original = positive_float(scaling, "original_max_position_embeddings");
    result.depth.rope_low_wave = original / positive_float(scaling, "low_freq_factor");
    result.depth.rope_high_wave = original / positive_float(scaling, "high_freq_factor");
    if (result.depth.rope_low_wave <= result.depth.rope_high_wave || original <= 0)
        throw std::invalid_argument("invalid llama3 RoPE frequency bounds");
    // Current fused language RMS operators retain at most 2048 values per row.
    for (const auto *cfg : {&result.text, &result.backbone, &result.depth})
        if (cfg->hidden > 2048)
            throw std::invalid_argument(
                "language hidden size exceeds the fused RMS kernel capacity");
    result.text.quantization = result.backbone.quantization = result.depth.quantization = quantization;
    return result;
}
} // namespace tts
