#include "models/gpt_sovits/speech_encoder.h"

namespace tts::gpt_sovits {
namespace {
AffineNorm norm(Weights &w, const std::string &name, float eps) {
    return {w.get(name + ".weight"), w.get(name + ".bias"), eps};
}
std::vector<EncoderBlock> hubert_blocks(Weights &w, const Json &c) {
    std::vector<EncoderBlock> result;
    if (c.at("do_stable_layer_norm").get<bool>())
        throw std::invalid_argument("this HuBERT checkpoint requires a pre-norm encoder");
    int hidden = c.at("hidden_size");
    Type type = w.entry("hubert.feature_projection.projection.weight").type;
    const bool quantize = w.backend().has_quantization();
    for (int i = 0; i < c.at("num_hidden_layers").get<int>(); ++i) {
        auto p = "hubert.encoder.layers." + std::to_string(i);
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
ops::ConvGeometry position_geometry(Weights &w, const Json &cfg) {
    int kernel = cfg.at("num_conv_pos_embeddings"),
        groups = cfg.at("num_conv_pos_embedding_groups");
    const auto &s = w.entry("hubert.encoder.pos_conv_embed.conv.weight").shape;
    int channels = cfg.at("hidden_size");
    if (groups < 1 || channels % groups ||
        s != std::vector<int>{channels, channels / groups, kernel})
        throw std::invalid_argument("HuBERT positional convolution mismatch");
    ops::ConvGeometry g{channels, channels, kernel, 1, kernel / 2, 1, groups, false};
    g.padding_right = (kernel - 1) / 2;
    return g;
}
} // namespace
Hubert::Hubert(Weights &w, const Json &cfg)
    : temporal_weight_(w.get("hubert.feature_extractor.conv_layers.0.layer_norm.weight")),
      temporal_bias_(w.get("hubert.feature_extractor.conv_layers.0.layer_norm.bias")),
      feature_norm_(norm(w, "hubert.feature_projection.layer_norm", cfg.at("layer_norm_eps"))),
      input_norm_(norm(w, "hubert.encoder.layer_norm", cfg.at("layer_norm_eps"))),
      projection_(w, "hubert.feature_projection.projection"),
      position_(w, "hubert.encoder.pos_conv_embed.conv", position_geometry(w, cfg)),
      encoder_(hubert_blocks(w, cfg), cfg.at("num_attention_heads")) {
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
        auto p = "hubert.feature_extractor.conv_layers." + std::to_string(i) + ".conv";
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
Hubert::Workspace::Workspace(const Hubert &h, int batch, int samples, BufferAllocator *allocator)
    : encoder(h.encoder_, batch, h.frames(samples), allocator), max_batch(batch),
      max_samples(samples) {
    int time = samples;
    size_t activation_bytes = 0, projection_bytes = 0;
    for (const auto &c : h.features_) {
        projection_bytes = std::max(projection_bytes, c.workspace_bytes(batch, time));
        time = c.geometry().output_length(time);
        activation_bytes =
            std::max(activation_bytes,
                     size_t(batch) * time * c.geometry().output_channels * bytes(c.type()));
    }
    projection_bytes = std::max(projection_bytes, h.position_.workspace_bytes(batch, time));
    activation_bytes = std::max(activation_bytes, size_t(batch) * time * h.encoder_.hidden() *
                                                      bytes(h.encoder_.type()));
    for (auto &a : activations)
        a = Buffer(activation_bytes, allocator);
    projected =
        Buffer(size_t(batch) * time * h.encoder_.hidden() * bytes(h.encoder_.type()), allocator);
    convolution = Buffer(projection_bytes, allocator);
    statistics =
        Buffer(size_t(batch) * h.features_.front().geometry().output_channels * 2 * sizeof(float),
               allocator);
}
TensorView Hubert::forward(Workspace &w, TensorView waveform, const ForwardBatch &input) const {
    auto &gpu = backend_for(waveform);
    int batch = input.batch(), samples = input.time();
    if (batch < 1 || batch > w.max_batch || samples < 1 || samples > w.max_samples ||
        waveform.rows != int64_t(batch) * samples || waveform.cols != 1)
        throw std::invalid_argument("HuBERT waveform exceeds initialized workspace");
    int time = samples;
    TensorView x = waveform;
    for (size_t i = 0; i < features_.size(); ++i) {
        const auto &c = features_[i];
        int output = c.geometry().output_length(time), channels = c.geometry().output_channels;
        auto y = w.activations[i % 2].view(batch * output, channels, c.type());
        ops::Epilogue e;
        if (i)
            e.activation = ops::Activation::Gelu;
        TensorView projection;
        if (c.workspace_bytes(batch, time))
            projection =
                w.convolution.view(batch * time, c.geometry().kernel * channels, Type::F32);
        c.forward(x, y, ForwardBatch{batch, time}, projection, e);
        if (!i)
            ops::temporal_norm(gpu.stream, y, y, temporal_weight_, temporal_bias_, batch, output,
                               1.e-5f, ops::Activation::Gelu,
                               w.statistics.view(batch, 2 * channels, Type::F32));
        time = output;
        x = y;
    }
    ops::layer_norm(gpu.stream, x, x, feature_norm_.weight, feature_norm_.bias,
                    feature_norm_.epsilon);
    auto projected = w.projected.view(batch * time, encoder_.hidden(), encoder_.type());
    projection_.forward(x, projected);
    auto position = w.activations[0].view(batch * time, encoder_.hidden(), encoder_.type());
    ops::Epilogue e;
    e.activation = ops::Activation::Gelu;
    position_.forward(projected, position, ForwardBatch{batch, time},
                      w.convolution.view(
                          batch * time, position_.geometry().kernel * encoder_.hidden(), Type::F32),
                      e);
    ops::NormFusion f;
    f.residual = projected;
    ops::layer_norm(gpu.stream, position, projected, input_norm_.weight, input_norm_.bias,
                    input_norm_.epsilon, f);
    encoder_.forward(w.encoder, projected, {batch, time});
    return projected;
}
} // namespace tts::gpt_sovits
