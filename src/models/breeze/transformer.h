#pragma once
#include <algorithm>
#include "models/breeze/weights.h"
#include "models/breeze/transformer_config.h"
#include "core/row_capacity.h"

namespace tts {
struct LayerRange {
    int begin = 0, end = -1;
};
struct TransformerLayer {
    TensorView pre, post, attn_norm, ffn_norm, qkv, out, gateup, down, qn, kn, attn_scale,
        mlp_scale;
    RowScaledView qkv_quant, gateup_quant, down_quant;
};
struct TransformerState {
    Buffer keys, values, normed, projected, query, attended, update, gateup, gated;
    Buffer quant_values, quant_scales;
    int batch, capacity, max_rows;
    KVCacheView external_kv;
    bool ring;
    TransformerState(const TransformerConfig &c, int b, int cap, int rows, BufferAllocator *arena,
                     KVCacheView cache = {}, bool circular = false, bool owns_cache = true)
        : keys((!owns_cache || cache.layout.block_table || cache.layout.sequences ||
                cache.layout.key_pages)
                   ? 0
                   : size_t(c.layers) * b * cap * c.kvheads * c.dim * bytes(c.type),
               arena),
          values(keys.size(), arena),
          normed(c.quantization != Quantization::None ? 0 : size_t(rows) * c.hidden * bytes(c.type),
                 arena),
          projected(size_t(rows) * (c.heads + 2 * c.kvheads) * c.dim * bytes(c.type), arena),
          query(size_t(rows) * c.heads * c.dim * bytes(c.type), arena),
          attended(query.size(), arena), update(size_t(rows) * c.hidden * bytes(c.type), arena),
          gateup(size_t(rows) * c.intermediate * 2 * bytes(c.type), arena),
          gated(c.quantization != Quantization::None ? 0 : gateup.size() / 2, arena),
          quant_values(c.quantization != Quantization::None
                           ? size_t(rows) * std::max(c.hidden, c.intermediate)
                           : 0,
                       arena),
          quant_scales(c.quantization != Quantization::None ? size_t(rows) * sizeof(float) : 0,
                       arena),
          batch(b), capacity(cap), max_rows(rows), external_kv(cache), ring(circular) {}
};
class Transformer {
  public:
    TransformerConfig config;
    std::vector<TransformerLayer> layers;
    TensorView final_norm;
    Tensor rope_storage[2];
    TensorView rope[2];
    Transformer(Weights &weights, const std::string &prefix, TransformerConfig cfg) : config(cfg) {
        for (int i = 0; i < cfg.layers; i++) {
            std::string p = prefix + ".layers." + std::to_string(i);
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
            l.attn_scale = weights.optional(p + ".self_attn_layer_scale.scale");
            l.mlp_scale = weights.optional(p + ".mlp_layer_scale.scale");
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
                Tensor({cfg.max_positions, cfg.dim}, cfg.type, &weights.backend().memory);
            rope[i] = rope_storage[i].view();
            weights.backend().make_rope(rope[i], cfg.dim, i ? cfg.full_theta : cfg.theta,
                                        i ? cfg.full_factor : cfg.factor, i ? 1 : cfg.rope_type,
                                        cfg.rope_low_wave, cfg.rope_high_wave);
        }
    }
    void prepare_range(Backend &gpu, int max_rows) const {
        if (config.quantization == Quantization::None)
            return;
        for (int rows = 1; rows <= max_rows;) {
            int capacity = row_capacity(rows);
            prepare(gpu, capacity);
            rows = capacity + 1;
        }
    }
    void prepare(Backend &gpu, int rows) const {
        if (config.quantization == Quantization::None)
            return;
        const auto &c = config;
        for (auto shape :
             {std::pair{(c.heads + 2 * c.kvheads) * c.dim, c.hidden},
              std::pair{2 * c.intermediate, c.hidden}, std::pair{c.hidden, c.intermediate}})
            gpu.quantized().prepare(rows, shape.first, shape.second);
    }
    void forward(Backend &gpu, TransformerState &s, TensorView x, const int *positions, int batch,
                 int time, int slot = 0, int fixed_position = -1, TensorView final_last = {},
                 const PackedBatch *packed = nullptr, const CacheSequence *sequences = nullptr,
                 LayerRange range = {}) const {
        const auto &c = config;
        int end = range.end < 0 ? c.layers : range.end;
        if (range.begin < 0 || range.begin >= end || end > c.layers)
            throw std::invalid_argument("invalid transformer layer range");
        int n = packed ? packed->rows : batch * time;
        if (batch < 1 || time < 1 || slot < 0 || n > s.max_rows || x.cols != c.hidden ||
            x.rows != n || slot + batch > s.batch ||
            (s.ring && (!c.causal || c.window <= 0 || s.capacity < c.window || time != 1)) ||
            (!s.ring && time > s.capacity) ||
            (!positions && !packed && !sequences &&
             (fixed_position < 0 || fixed_position + time > rope[0].rows ||
              (!s.ring && fixed_position + time > s.capacity))))
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
        size_t cache_stride = size_t(s.batch) * s.capacity * c.kvheads * c.dim * bytes(c.type);
        if (!range.begin) {
            if (quantized)
                gpu.quantized().norm(x, {}, layers[0].pre, quant_view(c.hidden), c.eps, c.gemma);
            else
                gpu.norm(x, layers[0].pre, norm, c.eps, c.gemma);
        }
        for (int i = range.begin; i < end; i++) {
            const auto &l = layers[i];
            if (quantized)
                gpu.quantized().linear(quant_view(c.hidden), l.qkv_quant, proj);
            else
                gpu.linear(norm, l.qkv, proj);
            const auto &external = s.external_kv;
            auto k = sequences && !external.layout.key_pages ? TensorView{}
                     : external.layout.key_pages             ? external.keys
                     : external.layout.block_table
                         ? external.keys.slice(i * external.rows_per_layer, external.rows_per_layer)
                         : s.keys.view(s.batch * s.capacity, c.kvheads * c.dim, c.type,
                                       i * cache_stride);
            auto v =
                sequences && !external.layout.key_pages ? TensorView{}
                : external.layout.key_pages             ? external.values
                : external.layout.block_table
                    ? external.values.slice(i * external.rows_per_layer, external.rows_per_layer)
                    : s.values.view(s.batch * s.capacity, c.kvheads * c.dim, c.type,
                                    i * cache_stride);
            CacheLayout cache = external.layout;
            cache.ring = s.ring;
            cache.sequences = sequences ? sequences : cache.sequences;
            cache.layer = i;
            bool full = c.gemma && c.full_attention[i];
            gpu.qkv(proj, q, k, v, positions, batch, time, slot, s.capacity, c.heads, c.kvheads,
                    c.dim, l.qn, l.kn, c.eps, c.gemma, rope[full ? 1 : 0], fixed_position, packed,
                    cache);
            gpu.attention(q, k, v, a, positions, batch, time, slot, s.capacity, c.heads, c.kvheads,
                          c.dim, c.causal, full ? 0 : c.window, fixed_position, packed, cache);
            gpu.linear(a, l.out, u);
            if (c.gemma)
                gpu.norm(u, l.attn_norm, u, c.eps, true);
            if (quantized) {
                gpu.quantized().norm(x, u, l.post, quant_view(c.hidden), c.eps, c.gemma,
                                     l.attn_scale);
                gpu.quantized().linear(quant_view(c.hidden), l.gateup_quant, gu);
                gpu.quantized().gated(gu, quant_view(c.intermediate), c.gemma ? 1 : 0);
                gpu.quantized().linear(quant_view(c.intermediate), l.down_quant, u);
            } else {
                gpu.residual_norm(x, u, l.post, norm, c.eps, l.attn_scale, c.gemma);
                gpu.linear(norm, l.gateup, gu);
                gpu.gated(gu, g, c.gemma ? 1 : 0);
                gpu.linear(g, l.down, u);
            }
            if (c.gemma)
                gpu.norm(u, l.ffn_norm, u, c.eps, true);
            if (i + 1 == c.layers && final_last.data && packed)
                gpu.residual_norm_last(x, u, final_norm, final_last, packed->offsets + 1, c.eps);
            else if (i + 1 == c.layers && final_last.data)
                gpu.residual_norm(x.slice(n - 1), u.slice(n - 1), final_norm, final_last, c.eps,
                                  l.mlp_scale, c.gemma);
            else if (quantized && i + 1 < c.layers)
                gpu.quantized().norm(x, u, layers[i + 1].pre, quant_view(c.hidden), c.eps, c.gemma,
                                     l.mlp_scale);
            else
                gpu.residual_norm(x, u, i + 1 < c.layers ? layers[i + 1].pre : final_norm,
                                  i + 1 < c.layers ? norm : x, c.eps, l.mlp_scale, c.gemma);
        }
    }
};
} // namespace tts
