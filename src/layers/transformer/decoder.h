#pragma once
#include "core/buffer.h"
#include <algorithm>
#include "io/weights.h"
#include "ops/basic.h"
#include "layers/transformer/decoder_config.h"
#include "core/row_capacity.h"
#include "core/forward_batch.h"
#include "core/forward_context.h"

namespace tts {
struct TransformerLayer {
    TensorView pre, post, attn_norm, ffn_norm, qkv, out, gateup, down, qn, kn, attn_scale,
        mlp_scale;
    RowScaledView qkv_quant, gateup_quant, down_quant;
};
class Transformer {
    struct Workspace {
        Buffer normed, projected, query, attended, update, gateup, gated;
        Buffer quant_values, quant_scales;
        int batch, capacity, max_rows;
        Workspace(const TransformerConfig &c, int b, int cap, int rows)
            : normed(c.quantization != Quantization::None ? 0 : size_t(rows) * c.hidden * bytes(c.type)),
              projected(size_t(rows) * (c.heads + 2 * c.kvheads) * c.dim * bytes(c.type)),
              query(size_t(rows) * c.heads * c.dim * bytes(c.type)),
              attended(query.size()), update(size_t(rows) * c.hidden * bytes(c.type)),
              gateup(size_t(rows) * c.intermediate * 2 * bytes(c.type)),
              gated(c.quantization != Quantization::None ? 0 : gateup.size() / 2),
              quant_values(c.quantization != Quantization::None
                               ? size_t(rows) * std::max(c.hidden, c.intermediate)
                               : 0),
              quant_scales(c.quantization != Quantization::None ? size_t(rows) * sizeof(float) : 0),
              batch(b), capacity(cap), max_rows(rows) {}
    };

  public:
    TransformerConfig config;
    std::vector<TransformerLayer> layers;
    TensorView final_norm;
    Tensor rope_storage[2];
    TensorView rope[2];
    Transformer(Weights &weights, const std::string &prefix, TransformerConfig cfg) : config(cfg) {
        for (int i = 0; i < cfg.layers; i++) {
            std::string p = prefix + ".layers." + std::to_string(i);
            weights.validate(p + ".self_attn.q_proj.weight", {cfg.heads * cfg.dim, cfg.hidden},
                             cfg.type);
            weights.validate(p + ".self_attn.k_proj.weight", {cfg.kvheads * cfg.dim, cfg.hidden},
                             cfg.type);
            weights.validate(p + ".self_attn.v_proj.weight", {cfg.kvheads * cfg.dim, cfg.hidden},
                             cfg.type);
            weights.validate(p + ".mlp.gate_proj.weight", {cfg.intermediate, cfg.hidden}, cfg.type);
            weights.validate(p + ".mlp.up_proj.weight", {cfg.intermediate, cfg.hidden}, cfg.type);
            weights.validate(p + ".mlp.down_proj.weight", {cfg.hidden, cfg.intermediate}, cfg.type);
            TransformerLayer l;
            l.pre = weights.get(
                p + (cfg.gemma ? ".pre_self_attn_layernorm.weight" : ".input_layernorm.weight"));
            l.post = weights.get(p + (cfg.gemma ? ".pre_feedforward_layernorm.weight"
                                                : ".post_attention_layernorm.weight"));
            if (cfg.gemma) {
                l.attn_norm = weights.get(p + ".post_self_attn_layernorm.weight");
                l.ffn_norm = weights.get(p + ".post_feedforward_layernorm.weight");
            }
            if (cfg.quantization != Quantization::None) {
                l.qkv_quant = weights.quantized(p + ".qkv.quant", {p + ".self_attn.q_proj.weight",
                                                                   p + ".self_attn.k_proj.weight",
                                                                   p + ".self_attn.v_proj.weight"});
                l.gateup_quant = weights.quantized(
                    p + ".gateup.quant", {p + ".mlp.gate_proj.weight", p + ".mlp.up_proj.weight"});
                l.down_quant = weights.quantized(p + ".down.quant", {p + ".mlp.down_proj.weight"});
            } else {
                l.qkv = weights.fused(p + ".qkv", {p + ".self_attn.q_proj.weight",
                                                   p + ".self_attn.k_proj.weight",
                                                   p + ".self_attn.v_proj.weight"});
                l.gateup = weights.fused(p + ".gateup",
                                         {p + ".mlp.gate_proj.weight", p + ".mlp.up_proj.weight"});
                l.down = weights.get(p + ".mlp.down_proj.weight");
            }
            l.out = weights.get(p + ".self_attn.o_proj.weight");
            if (cfg.qknorm) {
                l.qn = weights.get(p + ".self_attn.q_norm.weight");
                l.kn = weights.get(p + ".self_attn.k_norm.weight");
            }
            if (cfg.layer_scale) {
                l.attn_scale = weights.get(p + ".self_attn_layer_scale.scale");
                l.mlp_scale = weights.get(p + ".mlp_layer_scale.scale");
            }
            auto vector = [&](TensorView v, int width) {
                if (v.type != cfg.type || v.numel() != size_t(width))
                    throw std::invalid_argument(
                        "transformer vector weight/configuration mismatch: " + p);
            };
            auto matrix = [&](TensorView v, int rows, int cols) {
                if (v.rows != rows || v.cols != cols)
                    throw std::invalid_argument(
                        "transformer matrix weight/configuration mismatch: " + p);
            };
            vector(l.pre, cfg.hidden);
            vector(l.post, cfg.hidden);
            if (cfg.gemma) {
                vector(l.attn_norm, cfg.hidden);
                vector(l.ffn_norm, cfg.hidden);
            }
            if (cfg.qknorm) {
                vector(l.qn, cfg.dim);
                vector(l.kn, cfg.dim);
            }
            if (l.attn_scale.data)
                vector(l.attn_scale, cfg.hidden);
            if (l.mlp_scale.data)
                vector(l.mlp_scale, cfg.hidden);
            bool quantized = cfg.quantization != Quantization::None;
            matrix(quantized ? l.qkv_quant.values : l.qkv, (cfg.heads + 2 * cfg.kvheads) * cfg.dim,
                   cfg.hidden);
            matrix(l.out, cfg.hidden, cfg.heads * cfg.dim);
            if (l.out.type != cfg.type)
                throw std::invalid_argument("transformer output projection type mismatch: " + p);
            matrix(quantized ? l.gateup_quant.values : l.gateup, 2 * cfg.intermediate, cfg.hidden);
            matrix(quantized ? l.down_quant.values : l.down, cfg.hidden, cfg.intermediate);
            layers.push_back(l);
        }
        final_norm = weights.get(prefix + ".norm.weight");
        if (final_norm.type != cfg.type || final_norm.numel() != size_t(cfg.hidden))
            throw std::invalid_argument("transformer final norm weight/configuration mismatch");
        if (cfg.max_positions <= 0 ||
            (cfg.gemma && cfg.full_attention.size() != size_t(cfg.layers)))
            throw std::invalid_argument("invalid transformer position or layer configuration");
        for (int i = 0; i < (cfg.gemma ? 2 : 1); i++) {
            rope_storage[i] =
                Tensor({cfg.max_positions, cfg.dim}, cfg.type, weights.device());
            rope[i] = rope_storage[i].view();
            ops::rotary_table(rope[i], cfg.dim, i ? cfg.full_theta : cfg.theta,
                                        i ? cfg.full_factor : cfg.factor, i ? 1 : cfg.rope_type,
                                        cfg.rope_low_wave, cfg.rope_high_wave);
        }
    }
    TensorView forward(TensorView hidden_states, ForwardContext &context) const {
        Tensor::Scope scratch;
        if (context.mode == ForwardContext::Prepare) {
            Workspace workspace(config, 1, 1, 1);
            if (workspace.update.data()) {
                ops::reserve_attention(hidden_states.device,
                    size_t(context.max_batch) * context.max_time * config.heads);
                prepare(workspace, context.max_rows);
            }
            return context.batch.output.numel() ? context.batch.output : hidden_states;
        }
        auto attention = context.batch;
        auto &cache = attention.cache;
        if (!cache.layout.block_table && !cache.layout.sequences && !cache.layout.key_pages &&
            !cache.keys.numel() && !cache.values.numel()) {
            if (attention.is_decode()) throw std::invalid_argument("decode requires a cache");
            const int64_t layer_rows = int64_t(context.max_batch) * context.max_time;
            if (config.layers < 1 || layer_rows < 1 || layer_rows > INT32_MAX / config.layers)
                throw std::invalid_argument("transformer cache exceeds tensor dimensions");
            cache.rows_per_layer = int(layer_rows);
            cache.keys = context.tensor({int(layer_rows) * config.layers, config.kvheads * config.dim}, config.type);
            cache.values = context.tensor(cache.keys, config.type);
        }
        Workspace workspace(config, context.max_batch, context.max_time, context.max_rows);
        if (context.mode == ForwardContext::Execute)
            forward(workspace, hidden_states, attention);
        return context.batch.output.numel() ? context.batch.output : hidden_states;
    }
  private:
    void prepare_range(int max_rows) const {
        if (config.quantization == Quantization::None)
            return;
        for (int rows = 1; rows <= max_rows;) {
            int capacity = row_capacity(rows);
            prepare(capacity);
            rows = capacity + 1;
        }
    }
    void prepare(int rows) const {

        if (config.quantization == Quantization::None)
            return;
        const auto &c = config;
        for (auto shape :
             {std::pair{(c.heads + 2 * c.kvheads) * c.dim, c.hidden},
              std::pair{2 * c.intermediate, c.hidden}, std::pair{c.hidden, c.intermediate}})
            ops::prepare_quantized(layers.front().pre, rows, shape.first, shape.second, config.type);
    }
    void prepare(Workspace &workspace, int max_rows) const {
        prepare_range(max_rows);

        auto input = workspace.update.view(1, config.hidden, config.type);
        for (const auto &layer : layers) {
            if (config.quantization == Quantization::None) {
                ops::prepare_matmul(input, layer.qkv,
                    workspace.projected.view(1, layer.qkv.rows, config.type), false, 0, 1, 1, max_rows);
                ops::prepare_matmul(input, layer.gateup,
                    workspace.gateup.view(1, layer.gateup.rows, config.type), false, 0, 1, 1, max_rows);
                ops::prepare_matmul(workspace.gated.view(1, config.intermediate, config.type),
                    layer.down, input, false, 0, 1, 1, max_rows);
            }
            ops::prepare_matmul(workspace.attended.view(1, layer.out.cols, config.type),
                layer.out, input, false, 0, 1, 1, max_rows);
        }
    }
    void forward(Workspace &s, TensorView x, const ForwardBatch &input) const {

        const auto &c = config;
        int batch = input.batch(), time = input.time(), slot = input.slot;
        int fixed_position = input.fixed_position;
        const auto *positions = input.positions;
        const auto *packed = input.sequence.addressing();
        const auto *sequences = input.cache.layout.sequences;
        const bool ring = input.cache.layout.ring;
        const auto &cache_view = input.cache;
        if (!sequences && !cache_view.layout.key_pages &&
            (cache_view.rows_per_layer < 1 || cache_view.keys.cols != c.kvheads * c.dim ||
             cache_view.values.cols != c.kvheads * c.dim || cache_view.keys.type != c.type ||
             cache_view.values.type != c.type || int64_t(cache_view.rows_per_layer) * c.layers > cache_view.keys.rows ||
             int64_t(cache_view.rows_per_layer) * c.layers > cache_view.values.rows))
            throw std::invalid_argument("transformer cache shape/type mismatch");
        auto final_last = input.output;
        int n = input.rows();
        if (batch < 1 || time < 1 || slot < 0 || n > s.max_rows || x.cols != c.hidden ||
            x.rows != n || slot + batch > s.batch ||
            (ring && (!c.causal || c.window <= 0 || s.capacity < c.window || time != 1)) ||
            (!ring && time > s.capacity) ||
            (!positions && !packed && !sequences &&
             (fixed_position < 0 || fixed_position + time > rope[0].rows ||
              (!ring && fixed_position + time > s.capacity))))
            throw std::runtime_error("transformer workspace bounds");
        bool quantized = c.quantization != Quantization::None;
        auto norm = quantized ? TensorView{} : s.normed.view(n, c.hidden, c.type),
             proj = s.projected.view(n, (c.heads + 2 * c.kvheads) * c.dim, c.type);
        auto q = s.query.view(n, c.heads * c.dim, c.type),
             a = s.attended.view(n, c.heads * c.dim, c.type),
             u = s.update.view(n, c.hidden, c.type);
        auto gu = s.gateup.view(n, 2 * c.intermediate, c.type),
             g = quantized ? TensorView{} : s.gated.view(n, c.intermediate, c.type);
        auto quant_view = [&](int cols) -> RowScaledView {
            return {s.quant_values.view(n, cols, quantized_type(c.quantization)),
                    s.quant_scales.view(n, 1, Type::F32)};
        };
        if (quantized)
            ops::quantized_norm(x, {}, layers[0].pre, quant_view(c.hidden), c.eps, c.gemma);
        else
            ops::norm(x, layers[0].pre, norm, c.eps, c.gemma);
        for (int i = 0; i < c.layers; i++) {
            const auto &l = layers[i];
            if (quantized)
                ops::matmul(quant_view(c.hidden), l.qkv_quant, proj);
            else
                ops::matmul(norm, l.qkv, proj);
            const auto &external = input.cache;
            auto k = sequences && !external.layout.key_pages ? TensorView{}
                     : external.layout.key_pages ? external.keys
                     : external.keys.slice(i * external.rows_per_layer, external.rows_per_layer);
            auto v = sequences && !external.layout.key_pages ? TensorView{}
                     : external.layout.key_pages ? external.values
                     : external.values.slice(i * external.rows_per_layer, external.rows_per_layer);
            CacheLayout cache = external.layout;
            cache.ring = ring;
            cache.layer = i;
            bool full = c.gemma && c.full_attention[i];
            ops::write_qkv(proj, q, k, v, positions, batch, time, slot, s.capacity, c.heads, c.kvheads,
                    c.dim, l.qn, l.kn, c.eps, c.gemma, rope[full ? 1 : 0], fixed_position, packed,
                    cache);
            ops::attention(q, k, v, a, positions, batch, time, slot, s.capacity, c.heads, c.kvheads,
                          c.dim, c.causal, full ? 0 : c.window, fixed_position, packed, cache);
            ops::matmul(a, l.out, u);
            if (c.gemma)
                ops::norm(u, l.attn_norm, u, c.eps, true);
            if (quantized) {
                ops::quantized_norm(x, u, l.post, quant_view(c.hidden), c.eps, c.gemma,
                                     l.attn_scale);
                ops::matmul(quant_view(c.hidden), l.gateup_quant, gu);
                ops::quantized_gated(gu, quant_view(c.intermediate), c.gemma ? 1 : 0);
                ops::matmul(quant_view(c.intermediate), l.down_quant, u);
            } else {
                ops::residual_norm(x, u, l.post, norm, c.eps, l.attn_scale, c.gemma);
                ops::matmul(norm, l.gateup, gu);
                ops::gated_activation(gu, g, c.gemma ? 1 : 0);
                ops::matmul(g, l.down, u);
            }
            if (c.gemma)
                ops::norm(u, l.ffn_norm, u, c.eps, true);
            if (i + 1 == c.layers && final_last.data && packed)
                ops::residual_norm_last(x, u, final_norm, final_last, packed->cu_seqlens + 1, c.eps);
            else if (i + 1 == c.layers && final_last.data && final_last.rows == x.rows)
                ops::residual_norm(x, u, final_norm, final_last, c.eps, l.mlp_scale, c.gemma);
            else if (i + 1 == c.layers && final_last.data)
                ops::residual_norm(x.slice(n - 1), u.slice(n - 1), final_norm, final_last, c.eps,
                                  l.mlp_scale, c.gemma);
            else if (quantized && i + 1 < c.layers)
                ops::quantized_norm(x, u, layers[i + 1].pre, quant_view(c.hidden), c.eps, c.gemma,
                                     l.mlp_scale);
            else
                ops::residual_norm(x, u, i + 1 < c.layers ? layers[i + 1].pre : final_norm,
                                  i + 1 < c.layers ? norm : x, c.eps, l.mlp_scale, c.gemma);
        }
    }
};
} // namespace tts
