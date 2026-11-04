#include "ops/basic.h"
#include "layers/transformer/relative.h"
#include "ops/relative_attention.h"
namespace tts {
namespace {
ops::ConvGeometry conv(Weights &w, const std::string &p) {
    const auto &s = w.entry(p + ".weight").shape;
    if (s.size() != 3 || s[2] % 2 != 1)
        throw std::invalid_argument("relative encoder convolution geometry");
    return {s[1], s[0], s[2], 1, (s[2] - 1) / 2, 1, 1, false};
}
} // namespace
RelativeEncoder::RelativeEncoder(Weights &w, const std::string &prefix, int layers, int heads)
    : hidden_(w.entry(prefix + ".attn_layers.0.conv_q.weight").shape.at(0)), heads_(heads),
      type_(w.entry(prefix + ".attn_layers.0.conv_q.weight").type) {
    if (layers < 1 || heads < 1 || hidden_ % heads)
        throw std::invalid_argument("relative encoder dimensions");
    for (int i = 0; i < layers; ++i) {
        auto index = std::to_string(i), p = prefix + ".attn_layers." + index;
        for (const char *projection : {"conv_q", "conv_k", "conv_v"}) {
            auto name = p + "." + projection;
            w.validate(name + ".weight", {hidden_, hidden_, 1}, type_);
            w.validate(name + ".bias", {hidden_}, type_);
        }
        auto matrix = w.fused(p + ".qkv.weight",
                              {p + ".conv_q.weight", p + ".conv_k.weight", p + ".conv_v.weight"});
        auto bias =
            w.fused(p + ".qkv.bias", {p + ".conv_q.bias", p + ".conv_k.bias", p + ".conv_v.bias"});
        const auto &relative = w.entry(p + ".emb_rel_k");
        if (relative.type != type_ || relative.shape.size() != 3 || relative.shape[1] % 2 != 1 ||
            relative.shape[2] != hidden_ / heads ||
            (relative.shape[0] != 1 && relative.shape[0] != heads))
            throw std::invalid_argument("relative position checkpoint shape");
        w.validate(p + ".emb_rel_v", relative.shape, type_);
        auto norm = [&](const char *name) -> AffineNorm {
            auto n = prefix + name + index;
            return {w.get(n + ".gamma"), w.get(n + ".beta"), 1.e-5f};
        };
        auto rk = w.get(p + ".emb_rel_k")
                      .reshape({relative.shape[0] * relative.shape[1], relative.shape[2]});
        auto rv = w.get(p + ".emb_rel_v")
                      .reshape({relative.shape[0] * relative.shape[1], relative.shape[2]});
        auto a = prefix + ".ffn_layers." + index + ".conv_1",
             b = prefix + ".ffn_layers." + index + ".conv_2";
        blocks_.push_back({Linear(matrix, bias), Linear(w, p + ".conv_o"), rk, rv,
                           Convolution(w, a, conv(w, a)), Convolution(w, b, conv(w, b)),
                           norm(".norm_layers_1."), norm(".norm_layers_2."),
                           (relative.shape[1] - 1) / 2});
    }
}
RelativeEncoder::Workspace::Workspace(const RelativeEncoder &encoder, const ForwardContext &context)
    : max_batch(context.max_batch), max_time(context.max_time), max_rows(context.max_rows) {
    if (max_batch < 1 || max_time < 1 || max_rows < 1)
        throw std::invalid_argument("invalid relative encoder workspace");
    const size_t rows = max_rows;
    projected = Buffer(rows * encoder.hidden_ * 3 * bytes(encoder.type_));
    attended = Buffer(rows * encoder.hidden_ * bytes(encoder.type_));
    update = Buffer(attended.size());
    expanded = Buffer(rows * encoder.blocks_.front().first.geometry().output_channels * bytes(encoder.type_));
    size_t scratch_bytes = ops::relative_attention_workspace_bytes(
        1, max_rows, encoder.heads_, encoder.hidden_ / encoder.heads_, encoder.type_);
    for (const auto &block : encoder.blocks_) {
        scratch_bytes = std::max(scratch_bytes, block.first.workspace_bytes(1, max_rows));
        scratch_bytes = std::max(scratch_bytes, block.second.workspace_bytes(1, max_rows));
    }
    scratch = Buffer(scratch_bytes);
}
TensorView RelativeEncoder::forward(TensorView input, ForwardContext &context) const {
    Tensor::Scope scratch;
    Workspace workspace(*this, context);
    if (context.mode == ForwardContext::Prepare && input.data)
        prepare(workspace, context.max_rows);
    else if (context.mode == ForwardContext::Execute)
        forward(workspace, input, context.batch);
    return input;
}
void RelativeEncoder::prepare(Workspace &w, int rows) const {
    auto x = w.update.view(1, hidden_, type_);
    for (const auto &b : blocks_) {
        b.qkv.prepare_range(x, w.projected.view(1, 3 * hidden_, type_), rows);
        b.out.prepare_range(w.attended.view(1, hidden_, type_), x, rows);
        auto expanded = w.expanded.view(1, b.first.geometry().output_channels, type_);
        b.first.prepare(x, w.scratch.view(1, b.first.geometry().kernel * expanded.cols, Type::F32), rows);
        b.second.prepare(expanded, w.scratch.view(1, b.second.geometry().kernel * hidden_, Type::F32), rows);
    }
}
void RelativeEncoder::forward(Workspace &w, TensorView x,
                              const ForwardBatch &input) const {

    int batch = input.batch(), time = input.time();
    const auto *packed = input.sequence.addressing();
    if (batch < 1 || (!packed && batch > w.max_batch) || time < 1 || time > w.max_time ||
        x.rows != input.rows() ||
        x.rows > w.max_rows || x.cols != hidden_ || x.type != type_)
        throw std::invalid_argument("relative encoder execution bounds");
    int rows = x.rows;
    auto projected = w.projected.view(rows, 3 * hidden_, type_);
    auto attended = w.attended.view(rows, hidden_, type_),
         update = w.update.view(rows, hidden_, type_);
    for (const auto &b : blocks_) {
        b.qkv.project(x, projected);
        ops::relative_attention(projected, b.qkv.bias(), b.relative_k, b.relative_v,
                                attended, w.scratch.view(1, w.scratch.size() / sizeof(float), Type::F32),
                                batch, time, heads_, b.window, packed);
        b.out.project(attended, update);
        ops::NormFusion f;
        f.residual = x;
        f.projection_bias = b.out.bias();
        ops::layer_norm( update, x, b.attention_norm.weight, b.attention_norm.bias,
                        b.attention_norm.epsilon, f);
        auto expanded = w.expanded.view(rows, b.first.geometry().output_channels, type_);
        auto scratch = [&](const Convolution &c) {
            return w.scratch.view(rows, c.geometry().kernel * c.geometry().output_channels,
                                  Type::F32);
        };
        ops::Epilogue e;
        e.activation = ops::Activation::Relu;
        b.first.forward(x, expanded, input, scratch(b.first), e);
        b.second.forward(expanded, update, input, scratch(b.second));
        f.projection_bias = {};
        ops::layer_norm( update, x, b.feedforward_norm.weight, b.feedforward_norm.bias,
                        b.feedforward_norm.epsilon, f);
    }
}
} // namespace tts
