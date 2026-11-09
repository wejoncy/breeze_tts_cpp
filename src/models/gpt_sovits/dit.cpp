#include "models/gpt_sovits/dit.h"
#include "core/forward_context.h"
#include "ops/attention.h"
#include "ops/basic.h"
#include "ops/storage.h"
#include <cmath>

namespace tts {
namespace {
ops::ConvGeometry geometry(Weights &w, const std::string &name, int groups) {
    const auto &s = w.entry(name + ".weight").shape;
    if (s.size() != 3)
        throw std::invalid_argument("DiT convolution rank");
    return {s[1] * groups, s[0], s[2], 1, (s[2] - 1) / 2, 1, groups, false};
}
TensorView partition(Weights &w, const std::string &root, const std::string &name, int start, int count) {
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
DiT::DiT(Weights &w, const std::string &root, const Json &cfg, int max_time)
    : hidden_(cfg.at("hidden")), heads_(cfg.at("heads")), head_dim_(cfg.at("head_dim")),
      text_channels_(cfg.at("text_channels")), mel_channels_(cfg.at("mel_channels")),
      steps_(cfg.at("steps")), type_(w.entry(root + ".proj_out.weight").type),
      time_in_(w, root + ".time_embed.time_mlp.0"), time_out_(w, root + ".time_embed.time_mlp.2"),
      text_projection_(partition(w, root, "text", 2 * mel_channels_, text_channels_),
                       w.get(root + ".input_embed.proj.bias")),
      condition_projection_(partition(w, root, "condition", mel_channels_, mel_channels_)),
      noise_projection_(partition(w, root, "noise", 0, mel_channels_)),
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
    const bool quantize = (w.quantization() != Quantization::None);
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
                           Tensor({time_rows, 6 * hidden_}, type_, w.device())});
    }
    if (blocks_.empty())
        throw std::invalid_argument("empty DiT");
    for (const auto &block : blocks_)
        for (const auto *layer : {&block.qkv, &block.up, &block.down})
            quantization_capacity_.include(layer->workspace_capacity());
    for (const auto &block : text_blocks_)
        for (const auto *layer : {&block.up, &block.down})
            quantization_capacity_.include(layer->workspace_capacity());
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
            angle = ops::rounded(angle, type_);
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
    time_encoded_ = w.generated(root + ".times", time_table, time_rows, frequency_dim, type_);
    final_time_modulation_ = Tensor({time_rows, 2 * hidden_}, type_, w.device());
}
void DiT::prepare_weights() const {
    const int time_rows = time_encoded_.rows;
    std::vector<std::pair<Linear, TensorView>> modulations;
    for (const auto &block : blocks_)
        modulations.emplace_back(block.modulation, block.time_modulation.view());
    modulations.emplace_back(final_modulation_, final_time_modulation_.view());
    Tensor time_states({time_rows, hidden_}, type_, Device::cuda(time_encoded_.device));
    Tensor projected_states({time_rows, hidden_}, type_, Device::cuda(time_encoded_.device));
    ops::Epilogue activation;
    activation.activation = ops::Activation::Silu;
    auto transform = [&](const Linear &layer, TensorView input, TensorView output,
                         ops::Epilogue finish = {}) {
        ops::grouped_matmul(input, layer.dense_weight(), output, 1);
        finish.bias = layer.bias();
        ops::pointwise(output, output, finish);
    };
    for (auto [steps, offset] : sampling_offsets_) {
        auto input = time_states.view().slice(offset, steps);
        auto output = projected_states.view().slice(offset, steps);
        transform(time_in_, time_encoded_.slice(offset, steps), input, activation);
        transform(time_out_, input, output, activation);
        for (const auto &[layer, target] : modulations)
            transform(layer, output, target.slice(offset, steps));
    }
}
TensorView DiT::forward(const Condition &condition, ForwardContext &context,
                        bool need_negative) const {
    const int batch = context.batch.batch(), time = context.batch.time();
    const int rows = context.batch.rows(), branches = need_negative ? 2 : 1;
    if (batch < 1 || time < 1 || rows != int64_t(batch) * time || rows > INT32_MAX / branches)
        throw std::invalid_argument("invalid DiT conditioning dimensions");
    auto output = context.tensor({rows * branches, hidden_}, type_);
    const auto positive = output.slice(0, rows);
    const auto negative = need_negative ? output.slice(rows, rows) : TensorView{};
    Tensor::Scope scratch;
    auto text = context.tensor({rows, text_channels_}, type_);
    Buffer update_storage(size_t(rows) * std::max(text_channels_, hidden_) * bytes(type_));
    auto update = update_storage.view(rows, text_channels_, type_);
    const int expanded_width = text_blocks_.empty() ? text_channels_
                                                   : text_blocks_.front().up.output_channels();
    Buffer expanded_storage(size_t(rows) * expanded_width * bytes(type_));
    Buffer statistics(size_t(batch) * expanded_width * sizeof(float));
    Linear::Workspace linear(rows, quantization_capacity_);
    auto upload_rows = [&](TensorView source, Type type) {
        auto destination = context.tensor({rows, 1}, type);
        if (context.mode == ForwardContext::Execute && source.numel()) {
            if (source.rows != rows || source.cols != 1 || source.type != type ||
                source.device != -1 || !source.dense())
                throw std::invalid_argument("invalid DiT conditioning row metadata");
            destination.copy_(source, true);
        }
        return source.numel() ? destination : TensorView{};
    };
    auto feature_rows = upload_rows(condition.feature_rows, Type::I64);
    auto mel_row_map = upload_rows(condition.mel_rows, Type::I32);
    auto retained_feature_rows = upload_rows(condition.retained_feature_rows, Type::I64);
    if (context.mode == ForwardContext::Prepare && output.data) {
        text_projection_.prepare_range(text.slice(0), positive.slice(0), context.max_rows);
        auto mel = context.tensor({1, mel_channels_}, type_);
        auto projected = update_storage.view(1, hidden_, type_);
        condition_projection_.prepare_range(mel, projected, context.max_rows);
        ops::prepare_matmul(mel, condition_projection_.dense_weight(), projected, false, 1, 1,
                           1, context.max_rows);
        for (const auto &block : text_blocks_) {
            auto expanded = expanded_storage.view(1, block.up.output_channels(), type_);
            block.up.prepare_range(update.slice(0), expanded, context.max_rows);
            block.down.prepare_range(expanded, update.slice(0), context.max_rows);
        }
    }
    if (context.mode != ForwardContext::Execute) return output;
    if (!condition.feature_rows.numel() && (condition.features.rows != rows ||
        condition.features.cols != text_channels_ || condition.features.type != type_))
        throw std::invalid_argument("invalid DiT feature condition");
    int mel_rows = 0;
    for (auto mel : condition.mel) {
        if (mel.cols != mel_channels_ || mel.type != type_ || mel.rows > rows - mel_rows)
            throw std::invalid_argument("invalid DiT mel condition span");
        mel_rows += mel.rows;
    }
    if (!mel_rows || (!condition.mel_rows.numel() && mel_rows != rows))
        throw std::invalid_argument("invalid DiT mel condition rows");
    ops::positional_add( condition.features, text, text_positions_, time, 0, {}, feature_rows.ptr<const void *>(),
                        retained_feature_rows.ptr<void *>());
    for (const auto &block : text_blocks_) {
        block.depthwise.forward(text, update, ForwardBatch{batch, time}, {});
        auto expanded = expanded_storage.view(rows, block.up.output_channels(), type_);
        ops::Epilogue activation;
        activation.activation = ops::Activation::Gelu;
        if (!block.up.normalized_forward(linear, update,
                                      block.up.is_quantized() ? TensorView{} : update, expanded,
                                      block.norm, block.norm_bias, 1.e-6f, {}, activation)) {
            ops::layer_norm( update, update, block.norm, block.norm_bias, 1.e-6f);
            block.up.forward(linear, update, expanded, activation);
        }
        ops::global_response_norm( expanded, expanded, block.gamma, block.beta, batch, time,
                                  statistics.view(batch, expanded.cols, Type::F32));
        ops::Epilogue residual;
        residual.residual = text;
        block.down.project(linear, expanded, update);
        residual.bias = block.down.bias();
        ops::pointwise(update, text, residual);
    }
    if (need_negative)
        text_projection_.forward(text, negative);
    if (!mel_row_map.data) {
        text_projection_.forward(text, positive);
        int offset = 0;
        for (auto mel : condition.mel) {
            ops::matmul(mel, condition_projection_.dense_weight(), positive.slice(offset, mel.rows), false,
                       1.f);
            offset += mel.rows;
        }
    } else {
        auto projected = update_storage.view(mel_rows, hidden_, type_);
        int offset = 0;
        for (auto mel : condition.mel) {
            condition_projection_.project(mel, projected.slice(offset, mel.rows));
            offset += mel.rows;
        }
        ops::Epilogue epilogue;
        epilogue.residual = projected;
        epilogue.residual_rows = mel_row_map.ptr<int>();
        text_projection_.forward(text, positive, epilogue);
    }
    return output;
}
TensorView DiT::forward(TensorView noise, TensorView condition, TensorView velocity,
                        ForwardContext &context, int step, int steps) const {
    const int batch = context.batch.batch(), time = context.batch.time();
    const int input_rows = context.batch.rows();
    const int branches = input_rows > 0 ? velocity.rows / input_rows : 0;
    if (batch < 1 || time < 1 || step < 0 || step >= steps || !supports_steps(steps) ||
        input_rows != int64_t(batch) * time || noise.rows != input_rows || noise.cols != mel_channels_ ||
        (branches != 1 && branches != 2) || velocity.rows != int64_t(input_rows) * branches ||
        velocity.cols != mel_channels_ || noise.type != type_ || velocity.type != type_ ||
        condition.rows != velocity.rows || condition.cols != hidden_ || condition.type != type_)
        throw std::invalid_argument("invalid DiT forward dimensions");
    const int rows = velocity.rows, effective_batch = batch * branches;
    Tensor::Scope scratch;
    auto hidden_states = context.tensor({rows, hidden_}, type_);
    auto normalized = context.tensor({rows, hidden_}, type_);
    auto update = context.tensor({rows, hidden_}, type_);
    auto projected_qkv = context.tensor({rows, 3 * hidden_}, type_);
    auto attended = context.tensor({rows, hidden_}, type_);
    Buffer expanded_storage(size_t(rows) * blocks_.front().up.output_channels() * bytes(type_));
    auto lse = context.tensor({rows, heads_}, Type::F32);
    size_t convolution_bytes = 0;
    for (const auto &convolution : position_convs_)
        convolution_bytes = std::max(convolution_bytes, convolution.workspace_bytes(effective_batch, time));
    Buffer convolution_storage(convolution_bytes);
    Linear::Workspace linear(rows, quantization_capacity_);
    if (context.mode == ForwardContext::Prepare && velocity.data) {
        for (const auto &convolution : position_convs_)
            if (convolution.workspace_bytes(1, 1))
                convolution.prepare(hidden_states.slice(0),
                    convolution_storage.view(1, convolution.geometry().kernel * hidden_, Type::F32),
                    context.max_rows);
        noise_projection_.prepare_range(noise.slice(0), update.slice(0), context.max_rows);
        output_.prepare_range(normalized.slice(0), velocity.slice(0), context.max_rows);
        for (const auto &block : blocks_) {
            auto expanded = expanded_storage.view(1, block.up.output_channels(), type_);
            block.qkv.prepare_range(normalized.slice(0), projected_qkv.slice(0), context.max_rows);
            block.out.prepare_range(attended.slice(0), update.slice(0), context.max_rows);
            block.up.prepare_range(normalized.slice(0), expanded, context.max_rows, ops::Activation::GeluTanh);
            block.down.prepare_range(expanded, update.slice(0), context.max_rows);
        }
    }
    if (context.mode != ForwardContext::Execute) return velocity;
    const int time_index = sampling_offsets_.at(steps) + step;
    ops::Epilogue epilogue;
    auto projected = update.slice(0, input_rows);
    noise_projection_.project(noise, projected);
    for (int branch = 0; branch < branches; ++branch) {
        epilogue.residual = (branch ? condition.slice(input_rows, input_rows) : condition.slice(0, input_rows));
        ops::pointwise( projected, hidden_states.slice(branch * input_rows, input_rows), epilogue);
    }
    epilogue = {};
    epilogue.activation = ops::Activation::Mish;
    for (size_t index = 0; index < position_convs_.size(); ++index) {
        const auto &convolution = position_convs_[index];
        convolution.forward(index ? normalized : hidden_states, index ? update : normalized, ForwardBatch{effective_batch, time},
                  convolution_storage.view(rows, convolution.geometry().kernel * hidden_, Type::F32), epilogue);
    }
    // Keep each residual update in its projection buffer until the next normalization.
    // That normalization also materializes hidden_states for the following attention residual.
    ops::NormFusion pending_fusion;
    pending_fusion.residual = hidden_states;
    pending_fusion.rows_per_sequence = time;
    for (const auto &block : blocks_) {
        auto mod = block.time_modulation.view().slice(time_index);
        auto part = [&](int index) { return mod.columns(index * hidden_, hidden_); };
        auto fusion = pending_fusion;
        fusion.raw = hidden_states;
        fusion.shift = part(0);
        fusion.scale = part(1);
        if (block.qkv.is_quantized()) {
            auto input = block.qkv.normalize_input(linear, update, {}, {}, 1.e-6f, fusion);
            block.qkv.project(input, projected_qkv);
        } else {
            ops::layer_norm( update, normalized, {}, {}, 1.e-6f, fusion);
            block.qkv.project(linear, normalized, projected_qkv);
        }
        ops::qkv_rotary( projected_qkv, block.qkv.bias(), rotary_, time, heads_, head_dim_, head_dim_,
                        false);
        dense_attention( projected_qkv.columns(0, hidden_), projected_qkv.columns(hidden_, hidden_),
                    projected_qkv.columns(2 * hidden_, hidden_), attended, lse.ptr<float>(), effective_batch, time,
                    time, heads_, head_dim_);
        block.out.project(attended, update);
        fusion.residual = hidden_states;
        fusion.raw = hidden_states;
        fusion.gate = part(2);
        fusion.projection_bias = block.out.bias();
        fusion.shift = part(3);
        fusion.scale = part(4);
        auto expanded = expanded_storage.view(rows, block.up.output_channels(), type_);
        epilogue = {};
        epilogue.activation = ops::Activation::GeluTanh;
        auto down_input = block.down.is_quantized() ? block.down.quantized_view(linear, rows)
                                                : RowScaledView{};
        if (!block.up.normalized_forward(linear, update,
                                      block.up.is_quantized() ? TensorView{} : normalized, expanded,
                                      {}, {}, 1.e-6f, fusion, epilogue,
                                      nullptr, down_input)) {
            ops::layer_norm( update, normalized, {}, {}, 1.e-6f, fusion);
            block.up.forward(linear, normalized, expanded, epilogue, down_input);
        }
        // The next normalization consumes this projection with its residual.
        if (block.down.is_quantized())
            block.down.project(down_input, update);
        else
            block.down.project(linear, expanded, update);
        pending_fusion.projection_bias = block.down.bias();
        pending_fusion.gate = part(5);
    }
    auto mod = final_time_modulation_.view().slice(time_index);
    auto fusion = pending_fusion;
    fusion.scale = mod.columns(0, hidden_);
    fusion.shift = mod.columns(hidden_, hidden_);
    ops::layer_norm( update, normalized, {}, {}, 1.e-6f, fusion);
    output_.forward(normalized, velocity);
    return velocity;
}
} // namespace tts
