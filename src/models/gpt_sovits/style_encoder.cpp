#include "models/gpt_sovits/style_encoder.h"
#include "ops/attention.h"
#include <cmath>
namespace tts::gpt_sovits {
StyleEncoder::StyleEncoder(Weights &w, int heads, int temporal_layers)
    : first_(w, "acoustic.ref_enc.spectral.0.fc"), second_(w, "acoustic.ref_enc.spectral.3.fc"),
      qkv_(w.fused("acoustic.ref_enc.qkv.weight", {"acoustic.ref_enc.slf_attn.w_qs.weight",
                                                   "acoustic.ref_enc.slf_attn.w_ks.weight",
                                                   "acoustic.ref_enc.slf_attn.w_vs.weight"}),
           w.fused("acoustic.ref_enc.qkv.bias",
                   {"acoustic.ref_enc.slf_attn.w_qs.bias", "acoustic.ref_enc.slf_attn.w_ks.bias",
                    "acoustic.ref_enc.slf_attn.w_vs.bias"})),
      out_(w, "acoustic.ref_enc.slf_attn.fc"), final_(w, "acoustic.ref_enc.fc.fc"), heads_(heads) {
    int hidden = first_.output_channels();
    for (const char *projection : {"w_qs", "w_ks", "w_vs"}) {
        auto name = std::string("acoustic.ref_enc.slf_attn.") + projection;
        w.validate(name + ".weight", {hidden, hidden}, first_.activation_type());
        w.validate(name + ".bias", {hidden}, first_.activation_type());
    }
    if (heads < 1 || hidden % heads || temporal_layers < 1)
        throw std::invalid_argument("style attention geometry");
    for (int i = 0; i < temporal_layers; ++i) {
        auto p = "acoustic.ref_enc.temporal." + std::to_string(i) + ".conv1.conv";
        const auto &s = w.entry(p + ".weight").shape;
        if (s.size() != 3 || s[0] != 2 * hidden || s[1] != hidden || s[2] % 2 != 1)
            throw std::invalid_argument("style GLU checkpoint geometry");
        temporal_.emplace_back(
            w, p, ops::ConvGeometry{hidden, 2 * hidden, s[2], 1, (s[2] - 1) / 2, 1, 1, false});
    }
}
StyleEncoder::Workspace::Workspace(const StyleEncoder &s, int batch, int time,
                                   BufferAllocator *allocator)
    : max_batch(batch), max_time(time) {
    if (batch < 1 || time < 1)
        throw std::invalid_argument("invalid style workspace");
    size_t rows = size_t(batch) * time;
    Type type = s.first_.activation_type();
    int hidden_width = s.first_.output_channels();
    hidden = Buffer(rows * hidden_width * bytes(type), allocator);
    update = Buffer(hidden.size(), allocator);
    attended = Buffer(hidden.size(), allocator);
    gates = Buffer(hidden.size() * 2, allocator);
    projected = Buffer(hidden.size() * 3, allocator);
    embedding = Buffer(rows * s.final_.output_channels() * bytes(type), allocator);
    size_t scratch = 0;
    for (const auto &c : s.temporal_)
        scratch = std::max(scratch, c.workspace_bytes(batch, time));
    convolution = Buffer(scratch, allocator);
    lse = Buffer(rows * s.heads_ * sizeof(float), allocator);
}
void StyleEncoder::forward(Workspace &w, TensorView spectrum, TensorView style,
                           const ForwardBatch &input) const {
    auto &gpu = backend_for(spectrum);
    int batch = input.batch(), time = input.time();
    int hidden = first_.output_channels();
    Type type = first_.activation_type();
    if (batch < 1 || batch > w.max_batch || time < 1 || time > w.max_time ||
        spectrum.rows != int64_t(batch) * time || spectrum.cols != first_.input_channels() ||
        spectrum.type != type || style.rows != batch || style.cols != final_.output_channels() ||
        style.type != type)
        throw std::invalid_argument("style input exceeds initialized workspace");
    int rows = batch * time;
    auto x = w.hidden.view(rows, hidden, type), update = w.update.view(rows, hidden, type);
    auto gates = w.gates.view(rows, 2 * hidden, type),
         projected = w.projected.view(rows, 3 * hidden, type);
    ops::Epilogue e;
    e.activation = ops::Activation::Mish;
    first_.forward(spectrum, update, e);
    second_.forward(update, x, e);
    for (const auto &c : temporal_) {
        c.forward(x, gates, ForwardBatch{batch, time},
                  w.convolution.view(rows, c.geometry().kernel * 2 * hidden, Type::F32));
        ops::gated(gpu.stream, gates, x, ops::Activation::Identity, time, {}, x);
    }
    qkv_.forward(x, projected);
    auto attended = w.attended.view(rows, hidden, type);
    dense_attention(gpu.stream, projected.columns(0, hidden), projected.columns(hidden, hidden),
                projected.columns(2 * hidden, hidden), attended, w.lse.ptr<float>(), batch, time,
                time, heads_, hidden / heads_, false, 1.f / std::sqrt(float(hidden)));
    out_.project(attended, update);
    e = {};
    e.bias = out_.bias();
    e.residual = x;
    ops::pointwise(gpu.stream, update, x, e);
    auto embedding = w.embedding.view(rows, final_.output_channels(), type);
    final_.forward(x, embedding);
    ops::temporal_mean(gpu.stream, embedding, style, time);
}
} // namespace tts::gpt_sovits
