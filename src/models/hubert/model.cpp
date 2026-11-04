#include "ops/basic.h"
#include "models/hubert/model.h"

namespace tts {
namespace {
AffineNorm norm(Weights &w, const std::string &name, float eps) {
    return {w.get(name + ".weight"), w.get(name + ".bias"), eps};
}
std::vector<EncoderBlock> hubert_blocks(Weights &w, const std::string &prefix, const Json &c) {
    std::vector<EncoderBlock> result;
    if (c.at("do_stable_layer_norm").get<bool>())
        throw std::invalid_argument("this HuBERT checkpoint requires a pre-norm encoder");
    int hidden = c.at("hidden_size");
    Type type = w.entry(prefix + ".feature_projection.projection.weight").type;
    const bool quantize = (w.quantization() != Quantization::None);
    for (int i = 0; i < c.at("num_hidden_layers").get<int>(); ++i) {
        auto p = prefix + ".encoder.layers." + std::to_string(i);
        for (const char *projection : {"q_proj", "k_proj", "v_proj"}) {
            auto name = p + ".attention." + projection;
            w.validate(name + ".weight", {hidden, hidden}, type);
            w.validate(name + ".bias", {hidden}, type);
        }
        int intermediate = c.at("intermediate_size");
        w.validate(p + ".feed_forward.intermediate_dense.weight", {intermediate, hidden}, type);
        w.validate(p + ".feed_forward.output_dense.weight", {hidden, intermediate}, type);
        auto bias =
            w.fused(p + ".qkv.bias", {p + ".attention.q_proj.bias", p + ".attention.k_proj.bias",
                                      p + ".attention.v_proj.bias"});
        auto qkv = Linear::fused(w, p + ".qkv.weight",
                                {p + ".attention.q_proj.weight", p + ".attention.k_proj.weight",
                                 p + ".attention.v_proj.weight"}, bias, quantize);
        result.push_back({std::move(qkv), Linear(w, p + ".attention.out_proj"),
                          Linear(w, p + ".feed_forward.intermediate_dense", quantize),
                          Linear(w, p + ".feed_forward.output_dense", quantize),
                          norm(w, p + ".layer_norm", c.at("layer_norm_eps")),
                          norm(w, p + ".final_layer_norm", c.at("layer_norm_eps")),
                          ops::Activation::Gelu});
    }
    return result;
}
ops::ConvGeometry position_geometry(Weights &w, const std::string &prefix, const Json &cfg) {
    int kernel = cfg.at("num_conv_pos_embeddings"),
        groups = cfg.at("num_conv_pos_embedding_groups");
    const auto &s = w.entry(prefix + ".encoder.pos_conv_embed.conv.weight").shape;
    int channels = cfg.at("hidden_size");
    if (groups < 1 || channels % groups ||
        s != std::vector<int>{channels, channels / groups, kernel})
        throw std::invalid_argument("HuBERT positional convolution mismatch");
    ops::ConvGeometry g{channels, channels, kernel, 1, kernel / 2, 1, groups, false};
    g.padding_right = (kernel - 1) / 2;
    return g;
}
} // namespace
Hubert::Hubert(Weights &w, const std::string &prefix, const Json &cfg)
    : temporal_weight_(w.get(prefix + ".feature_extractor.conv_layers.0.layer_norm.weight")),
      temporal_bias_(w.get(prefix + ".feature_extractor.conv_layers.0.layer_norm.bias")),
      feature_norm_(norm(w, prefix + ".feature_projection.layer_norm", cfg.at("layer_norm_eps"))),
      input_norm_(norm(w, prefix + ".encoder.layer_norm", cfg.at("layer_norm_eps"))),
      projection_(w, prefix + ".feature_projection.projection"),
      position_(w, prefix + ".encoder.pos_conv_embed.conv", position_geometry(w, prefix, cfg)),
      encoder_(hubert_blocks(w, prefix, cfg), cfg.at("num_attention_heads")) {
    if (cfg.at("feat_extract_norm") != "group" || cfg.at("feat_extract_activation") != "gelu" ||
        cfg.at("hidden_act") != "gelu")
        throw std::invalid_argument("unsupported HuBERT feature configuration");
    auto channels = cfg.at("conv_dim").get<std::vector<int>>();
    auto kernels = cfg.at("conv_kernel").get<std::vector<int>>();
    auto strides = cfg.at("conv_stride").get<std::vector<int>>();
    if (channels.empty() || channels.size() != kernels.size() || channels.size() != strides.size())
        throw std::invalid_argument("invalid HuBERT feature structure");
    int input = 1;
    for (size_t i = 0; i < channels.size(); ++i) {
        auto p = prefix + ".feature_extractor.conv_layers." + std::to_string(i) + ".conv";
        features_.emplace_back(
            w, p, ops::ConvGeometry{input, channels[i], kernels[i], strides[i], 0, 1, 1, false},
            cfg.at("conv_bias").get<bool>() ? w.get(p + ".bias") : TensorView{});
        input = channels[i];
    }
}
int Hubert::frames(int samples) const {
    for (const auto &c : features_)
        samples = c.geometry().output_length(samples);
    return samples;
}
TensorView Hubert::forward(TensorView waveform, ForwardContext &context) const {
    const int batch = context.batch.batch(), samples = context.batch.time();
    if (batch < 1 || samples < 1 || waveform.rows != int64_t(batch) * samples ||
        waveform.cols != 1 || waveform.type != features_.front().type())
        throw std::invalid_argument("invalid HuBERT waveform shape/type");
    int time = samples;
    size_t activation_bytes = 0, projection_bytes = 0;
    for (const auto &convolution : features_) {
        projection_bytes = std::max(projection_bytes, convolution.workspace_bytes(batch, time));
        time = convolution.geometry().output_length(time);
        activation_bytes = std::max(activation_bytes,
            size_t(batch) * time * convolution.geometry().output_channels * bytes(convolution.type()));
    }
    projection_bytes = std::max(projection_bytes, position_.workspace_bytes(batch, time));
    activation_bytes = std::max(activation_bytes,
        size_t(batch) * time * encoder_.hidden() * bytes(encoder_.type()));
    auto projected = context.tensor({batch * time, encoder_.hidden()}, encoder_.type());
    Tensor::Scope scratch;
    Buffer activations[] = {Buffer(activation_bytes), Buffer(activation_bytes)};
    Buffer convolution_storage(projection_bytes);
    Buffer statistics(size_t(batch) * features_.front().geometry().output_channels * 2 * sizeof(float));
    time = samples;
    auto hidden_states = waveform;
    for (size_t layer_index = 0; layer_index < features_.size(); ++layer_index) {
        const auto &convolution = features_[layer_index];
        const auto geometry = convolution.geometry();
        const int output_time = geometry.output_length(time), channels = geometry.output_channels;
        auto output = activations[layer_index % 2].view(batch * output_time, channels, convolution.type());
        TensorView projection;
        if (convolution.workspace_bytes(batch, time))
            projection = convolution_storage.view(batch * time, geometry.kernel * channels, Type::F32);
        if (context.mode == ForwardContext::Prepare && projection.data)
            convolution.prepare(hidden_states.slice(0), projection.slice(0), batch * time);
        if (context.mode == ForwardContext::Execute) {
            ops::Epilogue activation;
            if (layer_index) activation.activation = ops::Activation::Gelu;
            convolution.forward(hidden_states, output, ForwardBatch{batch, time}, projection, activation);
            if (!layer_index)
                ops::temporal_norm(output, output, temporal_weight_, temporal_bias_, batch, output_time,
                    1.e-5f, ops::Activation::Gelu, statistics.view(batch, 2 * channels, Type::F32));
        }
        time = output_time;
        hidden_states = output;
    }
    if (context.mode == ForwardContext::Execute)
        ops::layer_norm(hidden_states, hidden_states, feature_norm_.weight, feature_norm_.bias,
                         feature_norm_.epsilon);
    auto encoder_context = context;
    encoder_context.batch = ForwardBatch{batch, time};
    encoder_context.max_batch = batch;
    encoder_context.max_time = time;
    encoder_context.max_rows = batch * time;
    projection_.forward(hidden_states, projected, encoder_context);
    auto position = activations[0].view(batch * time, encoder_.hidden(), encoder_.type());
    TensorView projection;
    if (position_.workspace_bytes(batch, time))
        projection = convolution_storage.view(batch * time,
            position_.geometry().kernel * encoder_.hidden(), Type::F32);
    if (context.mode == ForwardContext::Prepare && projection.data)
        position_.prepare(projected.slice(0), projection.slice(0), batch * time);
    if (context.mode == ForwardContext::Execute) {
        ops::Epilogue activation;
        activation.activation = ops::Activation::Gelu;
        position_.forward(projected, position, encoder_context.batch, projection, activation);
        ops::NormFusion normalization;
        normalization.residual = projected;
        ops::layer_norm(position, projected, input_norm_.weight, input_norm_.bias,
                         input_norm_.epsilon, normalization);
    }
    return encoder_.forward(projected, encoder_context);
}
} // namespace tts
