#include "layers/transformer.h"
#include "ops/attention.h"

namespace tts {
TransformerEncoder::TransformerEncoder(std::vector<EncoderBlock> blocks, int heads)
    : blocks_(std::move(blocks)),
      hidden_(blocks_.empty() ? 0 : blocks_.front().qkv.input_channels()), heads_(heads),
      type_(blocks_.empty() ? Type::BF16 : blocks_.front().qkv.activation_type()) {
    if (hidden_ < 1 || heads_ < 1 || hidden_ % heads_ ||
        (type_ != Type::BF16 && type_ != Type::F16))
        throw std::invalid_argument("invalid dense transformer geometry/type");
    int intermediate = blocks_.front().up.output_channels();
    for (const auto &b : blocks_) {
        if (b.qkv.input_channels() != hidden_ || b.qkv.output_channels() != hidden_ * 3 ||
            b.out.input_channels() != hidden_ || b.out.output_channels() != hidden_ ||
            b.up.input_channels() != hidden_ || b.up.output_channels() != intermediate ||
            b.down.input_channels() != intermediate || b.down.output_channels() != hidden_)
            throw std::invalid_argument("dense transformer block dimensions disagree");
        for (const auto *linear : {&b.qkv, &b.out, &b.up, &b.down}) {
            if (linear->activation_type() != type_)
                throw std::invalid_argument("dense transformer dtype mismatch");
            if (linear->is_quantized())
                quant_input_channels_ = std::max(quant_input_channels_, linear->input_channels());
        }
        for (const auto &n : {b.attention_norm, b.feedforward_norm})
            if (!n.weight.data || !n.bias.data || n.weight.type != type_ || n.bias.type != type_ ||
                n.weight.numel() != size_t(hidden_) || n.bias.numel() != size_t(hidden_) ||
                !(n.epsilon > 0))
                throw std::invalid_argument("dense transformer norm mismatch");
    }
}
TransformerEncoder::Workspace::Workspace(const TransformerEncoder &e, int batch, int time,
                                         BufferAllocator *allocator, int rows_capacity)
    : max_batch(batch), max_time(time), max_rows(rows_capacity) {
    if (batch < 1 || time < 1 || rows_capacity < 0 ||
        int64_t(batch) * time > INT32_MAX)
        throw std::invalid_argument("invalid encoder workspace capacity");
    if (!max_rows) max_rows = batch * time;
    if (max_rows > int64_t(batch) * time)
        throw std::invalid_argument("encoder row capacity exceeds sequence capacity");
    size_t rows = size_t(max_rows);
    auto storage = [&](int width) { return Buffer(rows * width * bytes(e.type_), allocator); };
    projected = storage(3 * e.hidden_);
    attended = storage(e.hidden_);
    update = storage(e.hidden_);
    expanded = storage(e.blocks_.front().up.output_channels());
    lse = Buffer(rows * e.heads_ * sizeof(float), allocator);
    linear = Linear::Workspace(max_rows, e.quant_input_channels_, allocator);
}
void TransformerEncoder::prepare(int max_rows) const {
    for (const auto &block : blocks_) {
        for (const auto *linear : {&block.qkv, &block.out, &block.down})
            linear->prepare_range(max_rows);
        block.up.prepare_range(max_rows, block.activation);
    }
}
void TransformerEncoder::forward(Workspace &w, TensorView x,
                                 const ForwardBatch &attention, int begin, int end) const {
    auto &gpu = backend_for(x);
    if (end < 0)
        end = int(blocks_.size());
    int batch = attention.batch(), time = attention.time(), prefix = attention.prefix;
    int rows = attention.rows();
    bool cached = attention.cache.layout.valid_paged();
    if (batch < 1 || batch > w.max_batch || time < 1 || time > w.max_time || prefix < 0 ||
        prefix > time || begin < 0 || end <= begin || end > int(blocks_.size()) || rows < 1 ||
        rows > w.max_rows || x.rows != rows || x.cols != hidden_ ||
        x.type != type_ || (cached && attention.capacity < time) ||
        (attention.output.data &&
         (!attention.output_rows || attention.output.rows < 1 ||
          attention.output.cols != hidden_ || attention.output.type != type_)) ||
        (attention.decode && (!cached || time != 1 || attention.sequence.addressing())))
        throw std::invalid_argument("dense encoder execution bounds");
    auto projected = w.projected.view(rows, 3 * hidden_, type_);
    auto attended = w.attended.view(rows, hidden_, type_);
    auto update = w.update.view(rows, hidden_, type_);
    bool projected_ready = false, input_ready = false;
    for (int i = begin; i < end; ++i) {
        const auto &b = blocks_[i];
        auto q = projected.columns(0, hidden_), k = projected.columns(hidden_, hidden_),
             v = projected.columns(2 * hidden_, hidden_);
        CacheLayout cache = attention.cache.layout;
        cache.layer = i;
        if (cached) {
            k = attention.cache.keys;
            v = attention.cache.values;
        }
        bool qkv_fused = projected_ready;
        projected_ready = false;
        if (!qkv_fused)
            qkv_fused = attention.decode && !b.qkv.is_quantized() &&
                gpu.linear_qkv(x, b.qkv.dense_weight(),
                               {q, k, v, b.qkv.bias(), cache, attention.positions,
                                hidden_, attention.capacity});
        if (!qkv_fused) {
            if (input_ready)
                b.qkv.project(b.qkv.quantized_view(w.linear, rows), projected);
            else
                b.qkv.project(w.linear, x, projected);
            if (cached) {
                gpu.qkv(projected, q, k, v, attention.positions, batch, time, 0,
                        attention.capacity, heads_, heads_, hidden_ / heads_, {}, {}, 0,
                        false, {}, 0, attention.sequence.addressing(), cache, b.qkv.bias());
            } else if (b.qkv.bias().data) {
                ops::pointwise(gpu.stream, projected, projected, {b.qkv.bias()});
            }
        }
        input_ready = false;
        auto attend = [&](TensorView query, TensorView output, int query_time, int key_time,
                          bool causal, DenseAttentionLayout layout = {}) {
            if (cached)
                prefill_attention(gpu.stream, query, k, v, output, w.lse.ptr<float>(), batch,
                              query_time, 0, attention.capacity, heads_, heads_, hidden_ / heads_,
                              causal, 0, nullptr, cache, key_time, 0, layout);
            else
            {
                if (!layout.sequences && attention.sequence.cu_seqlens) {
                    layout.cu_seqlens_q = attention.sequence.cu_seqlens;
                    layout.cu_seqlens_k = attention.sequence.cu_seqlens;
                }
                dense_attention(gpu.stream, query, k, v, output, w.lse.ptr<float>(), batch, query_time,
                            key_time, heads_, hidden_ / heads_, causal, 0, layout);
            }
        };
        if (attention.decode) {
            decode_attention(gpu.stream, q, k, v, attended, w.lse.ptr<float>(), attention.positions,
                         batch, heads_, heads_, cache, gpu.flash_decode_workspace(),
                         attention.decode_partitions);
        } else if (attention.bidirectional || attention.causal) {
            if (attention.bidirectional)
                attend(q, attended, time, time, false, {0, 0, attention.bidirectional});
            if (attention.causal)
                attend(q, attended, time, time, true, {0, 0, attention.causal});
        } else if (!prefix) {
            attend(q, attended, time, time, false);
        } else {
            DenseAttentionLayout layout{time, time};
            attend(q, attended, prefix, prefix, false, layout);
            if (prefix < time)
                attend(q.slice(prefix, rows - prefix), attended.slice(prefix, rows - prefix),
                       time - prefix, time, true, layout);
        }
        b.out.project(w.linear, attended, update);
        ops::NormFusion f;
        f.residual = x;
        f.projection_bias = b.out.bias();
        auto expanded = w.expanded.view(rows, b.up.output_channels(), type_);
        ops::Epilogue e;
        e.activation = b.activation;
        auto down_input = b.down.is_quantized() ? b.down.quantized_view(w.linear, rows)
                                                : RowScaledView{};
        auto residual = x;
        bool up_fused = b.up.normalized_forward(w.linear, update, attended, expanded,
                                                 b.attention_norm.weight, b.attention_norm.bias,
                                                 b.attention_norm.epsilon, f, e, nullptr, down_input);
        if (up_fused) {
            // Attention has consumed this buffer. Retain the normalized skip
            // here until the feedforward norm writes back to x.
            residual = attended;
        } else {
            ops::layer_norm(gpu.stream, update, x, b.attention_norm.weight, b.attention_norm.bias,
                            b.attention_norm.epsilon, f);
            b.up.forward(w.linear, x, expanded, e, down_input);
        }
        if (b.down.is_quantized())
            b.down.project(down_input, update);
        else
            b.down.project(w.linear, expanded, update);
        f.residual = residual;
        f.projection_bias = b.down.bias();
        bool selected = i + 1 == end && attention.output.data;
        f.source_rows = selected ? attention.output_rows : nullptr;
        if (attention.decode && up_fused && i + 1 < end) {
            auto next_cache = attention.cache.layout;
            next_cache.layer = i + 1;
            QkvProjectionOutput next{q, k, v, blocks_[i + 1].qkv.bias(), next_cache,
                                     attention.positions, hidden_, attention.capacity};
            projected_ready = blocks_[i + 1].qkv.normalized_forward(update, x, q, b.feedforward_norm.weight, b.feedforward_norm.bias,
                b.feedforward_norm.epsilon, f, {}, &next);
        }
        if (!projected_ready) {
            if (i + 1 < end && blocks_[i + 1].qkv.is_quantized()) {
                blocks_[i + 1].qkv.normalize_input(w.linear, update,
                                                    b.feedforward_norm.weight,
                                                    b.feedforward_norm.bias,
                                                    b.feedforward_norm.epsilon, f, x);
                input_ready = true;
            } else {
                ops::layer_norm(gpu.stream, update, selected ? attention.output : x,
                                b.feedforward_norm.weight, b.feedforward_norm.bias,
                                b.feedforward_norm.epsilon, f);
            }
        }
    }
}
} // namespace tts
