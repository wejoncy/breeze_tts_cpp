#include "ops/basic.h"
#include "layers/audio/style_encoder.h"
#include "core/forward_context.h"
#include "ops/attention.h"
#include <cmath>
namespace tts {
StyleEncoder::StyleEncoder(Weights &w, const std::string &prefix, int heads, int temporal_layers)
    : first_(w, prefix + ".spectral.0.fc"), second_(w, prefix + ".spectral.3.fc"),
      qkv_(w.fused(prefix + ".qkv.weight", {prefix + ".slf_attn.w_qs.weight",
                                                   prefix + ".slf_attn.w_ks.weight",
                                                   prefix + ".slf_attn.w_vs.weight"}),
           w.fused(prefix + ".qkv.bias",
                   {prefix + ".slf_attn.w_qs.bias", prefix + ".slf_attn.w_ks.bias",
                    prefix + ".slf_attn.w_vs.bias"})),
      out_(w, prefix + ".slf_attn.fc"), final_(w, prefix + ".fc.fc"), heads_(heads) {
    int hidden = first_.output_channels();
    for (const char *projection : {"w_qs", "w_ks", "w_vs"}) {
        auto name = std::string(prefix + ".slf_attn.") + projection;
        w.validate(name + ".weight", {hidden, hidden}, first_.activation_type());
        w.validate(name + ".bias", {hidden}, first_.activation_type());
    }
    if (heads < 1 || hidden % heads || temporal_layers < 1)
        throw std::invalid_argument("style attention geometry");
    for (int i = 0; i < temporal_layers; ++i) {
        auto p = prefix + ".temporal." + std::to_string(i) + ".conv1.conv";
        const auto &s = w.entry(p + ".weight").shape;
        if (s.size() != 3 || s[0] != 2 * hidden || s[1] != hidden || s[2] % 2 != 1)
            throw std::invalid_argument("style GLU checkpoint geometry");
        temporal_.emplace_back(
            w, p, ops::ConvGeometry{hidden, 2 * hidden, s[2], 1, (s[2] - 1) / 2, 1, 1, false});
    }
}
StyleEncoder::Workspace::Workspace(const StyleEncoder &s, int batch, int time)
    : max_batch(batch), max_time(time) {
    if (batch < 1 || time < 1)
        throw std::invalid_argument("invalid style workspace");
    size_t rows = size_t(batch) * time;
    Type type = s.first_.activation_type();
    int hidden_width = s.first_.output_channels();
    hidden = Buffer(rows * hidden_width * bytes(type));
    update = Buffer(hidden.size());
    attended = Buffer(hidden.size());
    gates = Buffer(hidden.size() * 2);
    projected = Buffer(hidden.size() * 3);
    embedding = Buffer(rows * s.final_.output_channels() * bytes(type));
    size_t scratch = 0;
    for (const auto &c : s.temporal_)
        scratch = std::max(scratch, c.workspace_bytes(batch, time));
    convolution = Buffer(scratch);
    lse = Buffer(rows * s.heads_ * sizeof(float));
}
TensorView StyleEncoder::forward(TensorView spectrum, ForwardContext &context) const {
    auto output = context.tensor({context.batch.batch(), final_.output_channels()}, first_.activation_type());
    Tensor::Scope scratch;
    Workspace workspace(*this, context.batch.batch(), context.batch.time());
    if (context.mode == ForwardContext::Prepare && output.data) prepare(workspace, spectrum);
    else if (context.mode == ForwardContext::Execute) forward(workspace, spectrum, output, context.batch);
    return output;
}
void StyleEncoder::prepare(Workspace &workspace, TensorView spectrum) const {
    const auto type = first_.activation_type();
    const int hidden = first_.output_channels(), rows = workspace.max_batch * workspace.max_time;
    auto states = workspace.hidden.view(1, hidden, type);
    auto update = workspace.update.view(1, hidden, type);
    first_.prepare_range(spectrum.slice(0), update, rows, ops::Activation::Mish);
    second_.prepare_range(update, states, rows, ops::Activation::Mish);
    for (const auto &convolution : temporal_)
        if (convolution.workspace_bytes(1, workspace.max_time))
            convolution.prepare(states, workspace.convolution.view(1,
                convolution.geometry().kernel * convolution.geometry().output_channels, Type::F32), rows);
    qkv_.prepare_range(states, workspace.projected.view(1, 3 * hidden, type), rows);
    out_.prepare_range(workspace.attended.view(1, hidden, type), update, rows);
    final_.prepare_range(states, workspace.embedding.view(1, final_.output_channels(), type), rows);
}
void StyleEncoder::forward(Workspace &w, TensorView spectrum, TensorView style,
                           const ForwardBatch &input) const {

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
        ops::gated( gates, x, ops::Activation::Identity, time, {}, x);
    }
    qkv_.forward(x, projected);
    auto attended = w.attended.view(rows, hidden, type);
    dense_attention( projected.columns(0, hidden), projected.columns(hidden, hidden),
                projected.columns(2 * hidden, hidden), attended, w.lse.ptr<float>(), batch, time,
                time, heads_, hidden / heads_, false, 1.f / std::sqrt(float(hidden)));
    out_.project(attended, update);
    e = {};
    e.bias = out_.bias();
    e.residual = x;
    ops::pointwise( update, x, e);
    auto embedding = w.embedding.view(rows, final_.output_channels(), type);
    final_.forward(x, embedding);
    ops::temporal_mean( embedding, style, time);
}
} // namespace tts
