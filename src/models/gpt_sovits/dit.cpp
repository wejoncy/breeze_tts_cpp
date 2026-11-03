#include "models/gpt_sovits/dit.h"
#include "ops/attention.h"
#include <cmath>

namespace tts::gpt_sovits {
namespace {
const std::string root = "acoustic.cfm.estimator";
ops::ConvGeometry geometry(Weights &w, const std::string &name, int groups) {
    const auto &s = w.entry(name + ".weight").shape;
    if (s.size() != 3)
        throw std::invalid_argument("DiT convolution rank");
    return {s[1] * groups, s[0], s[2], 1, (s[2] - 1) / 2, 1, groups, false};
}
TensorView partition(Weights &w, const std::string &name, int start, int count) {
    return w.matrix_columns(root + "." + name + ".packed", root + ".input_embed.proj.weight", start,
                            count);
}
std::vector<float> positions(int length, int dim) {
    std::vector<float> table(size_t(length) * dim);
    for (int p = 0; p < length; ++p)
        for (int i = 0; i < dim / 2; ++i) {
            float angle = p * std::pow(10000.f, -2.f * i / dim);
            table[size_t(p) * dim + i] = std::cos(angle);
            table[size_t(p) * dim + dim / 2 + i] = std::sin(angle);
        }
    return table;
}
} // namespace
DiT::DiT(Weights &w, const Json &cfg, int max_time)
    : hidden_(cfg.at("hidden")), heads_(cfg.at("heads")), head_dim_(cfg.at("head_dim")),
      text_channels_(cfg.at("text_channels")), mel_channels_(cfg.at("mel_channels")),
      steps_(cfg.at("steps")), type_(w.entry(root + ".proj_out.weight").type),
      time_in_(w, root + ".time_embed.time_mlp.0"), time_out_(w, root + ".time_embed.time_mlp.2"),
      text_projection_(partition(w, "text", 2 * mel_channels_, text_channels_),
                       w.get(root + ".input_embed.proj.bias")),
      condition_projection_(partition(w, "condition", mel_channels_, mel_channels_)),
      noise_projection_(partition(w, "noise", 0, mel_channels_)),
      final_modulation_(w, root + ".norm_out.linear"), output_(w, root + ".proj_out") {
    if (hidden_ < 1 || heads_ < 1 || head_dim_ < 1 || hidden_ != heads_ * head_dim_ ||
        text_channels_ < 2 || text_channels_ % 2 || mel_channels_ < 1 || steps_ < 1 || max_time < 1)
        throw std::invalid_argument("invalid DiT configuration");
    int time_rows = 0;
    for (int steps : FlowSampling::step_counts) {
        sampling_offsets_.emplace(steps, time_rows);
        time_rows += steps;
    }
    if (!supports_steps(steps_))
        throw std::invalid_argument("unsupported V5 Euler step count");
    w.validate(root + ".input_embed.proj.weight",
               {hidden_, 2 * mel_channels_ + text_channels_}, type_);
    const bool quantize = w.backend().has_quantization();
    for (int i = 0; i < cfg.at("text_layers").get<int>(); ++i) {
        auto p = root + ".text_embed.text_blocks." + std::to_string(i);
        text_blocks_.push_back(
            {Convolution(w, p + ".dwconv", geometry(w, p + ".dwconv", text_channels_)),
             w.get(p + ".norm.weight"), w.get(p + ".norm.bias"), w.get(p + ".grn.gamma"),
             w.get(p + ".grn.beta"), Linear(w, p + ".pwconv1", quantize),
             Linear(w, p + ".pwconv2", quantize)});
    }
    for (int i : {0, 2}) {
        auto p = root + ".input_embed.conv_pos_embed.conv1d." + std::to_string(i);
        const auto &s = w.entry(p + ".weight").shape;
        if (s.size() != 3 || s[1] < 1 || hidden_ % s[1])
            throw std::invalid_argument("DiT position convolution groups");
        position_convs_.emplace_back(w, p, geometry(w, p, hidden_ / s[1]));
    }
    for (int i = 0; i < cfg.at("layers").get<int>(); ++i) {
        auto p = root + ".transformer_blocks." + std::to_string(i);
        for (const char *projection : {"to_q", "to_k", "to_v"}) {
            auto name = p + ".attn." + projection;
            w.validate(name + ".weight", {hidden_, hidden_}, type_);
            w.validate(name + ".bias", {hidden_}, type_);
        }
        auto bias = w.fused(p + ".qkv.bias",
                            {p + ".attn.to_q.bias", p + ".attn.to_k.bias", p + ".attn.to_v.bias"});
        auto qkv = Linear::fused(w, p + ".qkv.weight",
                                {p + ".attn.to_q.weight", p + ".attn.to_k.weight",
                                 p + ".attn.to_v.weight"}, bias, quantize);
        blocks_.push_back({Linear(w, p + ".attn_norm.linear"), std::move(qkv),
                           Linear(w, p + ".attn.to_out.0"), Linear(w, p + ".ff.ff.0.0", quantize),
                           Linear(w, p + ".ff.ff.2", quantize),
                           Tensor({time_rows, 6 * hidden_}, type_, &w.backend().memory)});
    }
    if (blocks_.empty())
        throw std::invalid_argument("empty DiT");
    text_positions_ = w.generated(root + ".text_positions", positions(max_time, text_channels_),
                                  max_time, text_channels_, type_);
    // Upstream rotates the first head_dim features before reshaping into heads.
    // It does not rotate each head independently.
    auto frequencies = w.host_values(root + ".rotary_embed.inv_freq");
    if (frequencies.size() != size_t(head_dim_ / 2))
        throw std::invalid_argument("DiT rotary frequency shape");
    std::vector<float> rotary(size_t(max_time) * head_dim_);
    for (int p = 0; p < max_time; ++p)
        for (int i = 0; i < head_dim_ / 2; ++i) {
            float angle = p * frequencies[i];
            if (type_ == Type::F16)
                angle = float(__float2half_rn(angle));
            if (type_ == Type::BF16)
                angle = float(__float2bfloat16_rn(angle));
            rotary[size_t(p) * head_dim_ + i] = std::cos(angle);
            rotary[size_t(p) * head_dim_ + head_dim_ / 2 + i] = std::sin(angle);
        }
    rotary_ = w.generated(root + ".rotary", rotary, max_time, head_dim_, type_);
    int frequency_dim = time_in_.input_channels();
    std::vector<float> time_table(size_t(time_rows) * frequency_dim);
    for (auto [steps, offset] : sampling_offsets_)
        for (int step = 0; step < steps; ++step)
            for (int i = 0; i < frequency_dim / 2; ++i) {
                float angle = (1000.f * step / steps) *
                              std::exp(-std::log(10000.f) * i / (frequency_dim / 2 - 1));
                time_table[size_t(offset + step) * frequency_dim + i] = std::sin(angle);
                time_table[size_t(offset + step) * frequency_dim + frequency_dim / 2 + i] =
                    std::cos(angle);
            }
    auto encoded = w.generated(root + ".times", time_table, time_rows, frequency_dim, type_);
    final_time_modulation_ = Tensor({time_rows, 2 * hidden_}, type_, &w.backend().memory);
    std::vector<std::pair<Linear, TensorView>> modulations;
    for (const auto &block : blocks_)
        modulations.emplace_back(block.modulation, block.time_modulation.view());
    modulations.emplace_back(final_modulation_, final_time_modulation_.view());
    // Capture views, not layer addresses: registering a layer must not depend
    // on whether its owner later moves it while constructing the network.
    w.after_loading([encoded, first = time_in_, second = time_out_, time_rows,
                     ranges = sampling_offsets_, hidden = hidden_, type = type_,
                     modulations = std::move(modulations)](Backend &gpu) {
        Tensor t({time_rows, hidden}, type, &gpu.memory);
        Tensor projected({time_rows, hidden}, type, &gpu.memory);
        ops::Epilogue e;
        e.activation = ops::Activation::Silu;
        // These are derived parameters. Materialize them with matrix GEMM,
        // independently of the runtime vector-projection dispatch policy.
        auto transform = [&](const Linear &layer, TensorView x, TensorView y,
                             ops::Epilogue finish = {}) {
            gpu.grouped_linear(x, layer.dense_weight(), y, 1);
            finish.bias = layer.bias();
            ops::pointwise(gpu.stream, y, y, finish);
        };
        for (auto [steps, offset] : ranges) {
            auto input = t.view().slice(offset, steps);
            auto output = projected.view().slice(offset, steps);
            transform(first, encoded.slice(offset, steps), input, e);
            transform(second, input, output, e);
            for (const auto &[layer, target] : modulations)
                transform(layer, output, target.slice(offset, steps));
        }
    });
}
DiT::Workspace::Workspace(const DiT &d, int batch, int time, BufferAllocator *allocator)
    : max_batch(batch), max_time(time) {
    if (batch < 1 || time < 1)
        throw std::invalid_argument("invalid DiT workspace");
    size_t rows = size_t(batch) * time;
    auto tensor = [&](int width, Type type) {
        return Buffer(rows * width * bytes(type), allocator);
    };
    condition = tensor(d.hidden_, d.type_);
    negative_condition = tensor(d.hidden_, d.type_);
    x = tensor(d.hidden_, d.type_);
    normalized = tensor(d.hidden_, d.type_);
    update = tensor(d.hidden_, d.type_);
    qkv = tensor(3 * d.hidden_, d.type_);
    attended = tensor(d.hidden_, d.type_);
    int expanded_width = d.blocks_.front().up.output_channels();
    expanded = tensor(expanded_width, d.type_);
    text = tensor(d.text_channels_, d.type_);
    text_update = tensor(d.text_channels_, d.type_);
    int text_width =
        d.text_blocks_.empty() ? d.text_channels_ : d.text_blocks_.front().up.output_channels();
    text_expanded = tensor(text_width, d.type_);
    statistics = Buffer(size_t(batch) * text_width * sizeof(float), allocator);
    lse = Buffer(rows * d.heads_ * sizeof(float), allocator);
    size_t projection = 0;
    for (const auto &c : d.position_convs_)
        projection = std::max(projection, c.workspace_bytes(batch, time));
    convolution = Buffer(projection, allocator);
    int quant_input = 0;
    auto consider = [&](const Linear &layer) {
        if (layer.is_quantized()) quant_input = std::max(quant_input, layer.input_channels());
    };
    for (const auto &block : d.blocks_)
        for (const auto *layer : {&block.qkv, &block.up, &block.down}) consider(*layer);
    for (const auto &block : d.text_blocks_)
        for (const auto *layer : {&block.up, &block.down}) consider(*layer);
    if (rows > INT32_MAX) throw std::invalid_argument("DiT row capacity exceeds tensor dimensions");
    linear = Linear::Workspace(int(rows), quant_input, allocator);
}
void DiT::prepare(Workspace &w, TensorView mel_input, TensorView mel_output,
                  int max_rows) const {
    auto &gpu = backend_for(mel_input);
    if (max_rows < 1)
        throw std::invalid_argument("invalid DiT planning capacity");
    for (const auto &c : position_convs_)
        if (c.workspace_bytes(1, 1))
            c.prepare(w.x.view(1, c.geometry().input_channels, c.type()),
                      w.convolution.view(1, c.geometry().kernel * c.geometry().output_channels,
                                         Type::F32), max_rows);
    auto prepare = [&](const Linear &layer, Buffer &input, Buffer &output,
                       ops::Activation activation = ops::Activation::Identity) {
        layer.prepare_range(input.view(1, layer.input_channels(), type_),
                            output.view(1, layer.output_channels(), type_), max_rows, activation);
    };
    prepare(text_projection_, w.text, w.condition);
    auto projected = w.update.view(1, hidden_, type_);
    condition_projection_.prepare_range(mel_input, projected, max_rows);
    gpu.prepare_linear(mel_input, condition_projection_.dense_weight(), projected, false, 1, 1,
                       1, max_rows);
    noise_projection_.prepare_range(mel_input, projected, max_rows);
    output_.prepare_range(w.normalized.view(1, hidden_, type_), mel_output, max_rows);
    for (const auto &block : blocks_) {
        prepare(block.qkv, w.normalized, w.qkv);
        prepare(block.out, w.attended, w.update);
        prepare(block.up, w.normalized, w.expanded, ops::Activation::GeluTanh);
        prepare(block.down, w.expanded, w.update);
    }
    for (const auto &block : text_blocks_) {
        prepare(block.up, w.text_update, w.text_expanded);
        prepare(block.down, w.text_expanded, w.text_update);
    }
}
void DiT::prepare_condition(Workspace &w, const Condition &c, const ForwardBatch &input,
                            bool need_negative) const {
    auto &gpu = backend_for(w.text.device());
    int batch = input.batch(), time = input.time();
    if (batch < 1 || batch > w.max_batch || time < 1 || time > w.max_time ||
        (!c.feature_rows && (c.features.rows != batch * time || c.features.cols != text_channels_ ||
                             c.features.type != type_)))
        throw std::invalid_argument("DiT condition exceeds initialized workspace");
    int rows = batch * time;
    int mel_rows = 0;
    for (auto mel : c.mel) {
        if (mel.cols != mel_channels_ || mel.type != type_ || mel.rows > rows - mel_rows)
            throw std::invalid_argument("invalid DiT mel condition span");
        mel_rows += mel.rows;
    }
    if (!mel_rows || (!c.mel_rows && mel_rows != rows))
        throw std::invalid_argument("invalid DiT mel condition rows");
    auto text = w.text.view(rows, text_channels_, type_);
    auto update = w.text_update.view(rows, text_channels_, type_);
    ops::positional_add(gpu.stream, c.features, text, text_positions_, time, 0, {}, c.feature_rows,
                        c.retained_feature_rows);
    for (const auto &b : text_blocks_) {
        b.depthwise.forward(text, update, ForwardBatch{batch, time}, {});
        auto expanded = w.text_expanded.view(rows, b.up.output_channels(), type_);
        ops::Epilogue e;
        e.activation = ops::Activation::Gelu;
        if (!b.up.normalized_forward(w.linear, update,
                                      b.up.is_quantized() ? TensorView{} : update, expanded,
                                      b.norm, b.norm_bias, 1.e-6f, {}, e)) {
            ops::layer_norm(gpu.stream, update, update, b.norm, b.norm_bias, 1.e-6f);
            b.up.forward(w.linear, update, expanded, e);
        }
        ops::global_response_norm(gpu.stream, expanded, expanded, b.gamma, b.beta, batch, time,
                                  w.statistics.view(batch, expanded.cols, Type::F32));
        e = {};
        e.residual = text;
        b.down.project(w.linear, expanded, update);
        e.bias = b.down.bias();
        ops::pointwise(gpu.stream, update, text, e);
    }
    auto positive = w.condition.view(rows, hidden_, type_);
    auto negative = w.negative_condition.view(rows, hidden_, type_);
    if (need_negative)
        text_projection_.forward(text, negative);
    if (!c.mel_rows) {
        text_projection_.forward(text, positive);
        int offset = 0;
        for (auto mel : c.mel) {
            gpu.linear(mel, condition_projection_.dense_weight(), positive.slice(offset, mel.rows), false,
                       1.f);
            offset += mel.rows;
        }
    } else {
        auto projected = w.update.view(mel_rows, hidden_, type_);
        int offset = 0;
        for (auto mel : c.mel) {
            condition_projection_.project(mel, projected.slice(offset, mel.rows));
            offset += mel.rows;
        }
        ops::Epilogue e;
        e.residual = projected;
        e.residual_rows = c.mel_rows;
        text_projection_.forward(text, positive, e);
    }
}
void DiT::forward(Workspace &w, TensorView noise, TensorView velocity, const ForwardBatch &input,
                  int step, int steps) const {
    auto &gpu = backend_for(noise);
    int batch = input.batch(), time = input.time();
    int branches = noise.rows > 0 ? velocity.rows / noise.rows : 0;
    if (batch < 1 || batch > w.max_batch || time < 1 || time > w.max_time || step < 0 ||
        step >= steps || !supports_steps(steps) || noise.rows != int64_t(batch) * time || noise.cols != mel_channels_ ||
        (branches != 1 && branches != 2) || velocity.rows != int64_t(noise.rows) * branches ||
        batch > w.max_batch / branches || velocity.cols != mel_channels_ || noise.type != type_ ||
        velocity.type != type_)
        throw std::invalid_argument("DiT execution exceeds initialized workspace");
    int time_index = sampling_offsets_.at(steps) + step;
    int input_rows = batch * time;
    batch *= branches;
    int rows = batch * time;
    auto x = w.x.view(rows, hidden_, type_), norm = w.normalized.view(rows, hidden_, type_);
    auto update = w.update.view(rows, hidden_, type_), qkv = w.qkv.view(rows, 3 * hidden_, type_);
    auto attended = w.attended.view(rows, hidden_, type_);
    ops::Epilogue e;
    auto projected = update.slice(0, input_rows);
    noise_projection_.project(noise, projected);
    for (int branch = 0; branch < branches; ++branch) {
        e.residual = (branch ? w.negative_condition : w.condition).view(input_rows, hidden_, type_);
        ops::pointwise(gpu.stream, projected, x.slice(branch * input_rows, input_rows), e);
    }
    e = {};
    e.activation = ops::Activation::Mish;
    for (size_t i = 0; i < position_convs_.size(); ++i) {
        const auto &c = position_convs_[i];
        c.forward(i ? norm : x, i ? update : norm, ForwardBatch{batch, time},
                  w.convolution.view(rows, c.geometry().kernel * hidden_, Type::F32), e);
    }
    // Keep each residual update in its projection buffer until the next norm.
    // That norm also materializes x for the following attention residual.
    ops::NormFusion pending;
    pending.residual = x;
    pending.rows_per_sequence = time;
    for (const auto &b : blocks_) {
        auto mod = b.time_modulation.view().slice(time_index);
        auto part = [&](int i) { return mod.columns(i * hidden_, hidden_); };
        auto f = pending;
        f.raw = x;
        f.shift = part(0);
        f.scale = part(1);
        if (b.qkv.is_quantized()) {
            auto input = b.qkv.normalize_input(w.linear, update, {}, {}, 1.e-6f, f);
            b.qkv.project(input, qkv);
        } else {
            ops::layer_norm(gpu.stream, update, norm, {}, {}, 1.e-6f, f);
            b.qkv.project(w.linear, norm, qkv);
        }
        ops::qkv_rotary(gpu.stream, qkv, b.qkv.bias(), rotary_, time, heads_, head_dim_, head_dim_,
                        false);
        dense_attention(gpu.stream, qkv.columns(0, hidden_), qkv.columns(hidden_, hidden_),
                    qkv.columns(2 * hidden_, hidden_), attended, w.lse.ptr<float>(), batch, time,
                    time, heads_, head_dim_);
        b.out.project(attended, update);
        f.residual = x;
        f.raw = x;
        f.gate = part(2);
        f.projection_bias = b.out.bias();
        f.shift = part(3);
        f.scale = part(4);
        auto expanded = w.expanded.view(rows, b.up.output_channels(), type_);
        e = {};
        e.activation = ops::Activation::GeluTanh;
        auto down_input = b.down.is_quantized() ? b.down.quantized_view(w.linear, rows)
                                                : RowScaledView{};
        if (!b.up.normalized_forward(w.linear, update,
                                      b.up.is_quantized() ? TensorView{} : norm, expanded,
                                      {}, {}, 1.e-6f, f, e,
                                      nullptr, down_input)) {
            ops::layer_norm(gpu.stream, update, norm, {}, {}, 1.e-6f, f);
            b.up.forward(w.linear, norm, expanded, e, down_input);
        }
        // The next normalization consumes this projection with its residual.
        if (b.down.is_quantized())
            b.down.project(down_input, update);
        else
            b.down.project(w.linear, expanded, update);
        pending.projection_bias = b.down.bias();
        pending.gate = part(5);
    }
    auto mod = final_time_modulation_.view().slice(time_index);
    auto f = pending;
    f.scale = mod.columns(0, hidden_);
    f.shift = mod.columns(hidden_, hidden_);
    ops::layer_norm(gpu.stream, update, norm, {}, {}, 1.e-6f, f);
    output_.forward(norm, velocity);
}
} // namespace tts::gpt_sovits
