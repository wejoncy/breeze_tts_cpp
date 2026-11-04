#include "ops/basic.h"
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
    for (const auto &block : blocks_) {
        if (block.qkv.input_channels() != hidden_ || block.qkv.output_channels() != hidden_ * 3 ||
            block.out.input_channels() != hidden_ || block.out.output_channels() != hidden_ ||
            block.up.input_channels() != hidden_ || block.up.output_channels() != intermediate ||
            block.down.input_channels() != intermediate || block.down.output_channels() != hidden_)
            throw std::invalid_argument("dense transformer block dimensions disagree");
        for (const auto *linear : {&block.qkv, &block.out, &block.up, &block.down}) {
            if (linear->activation_type() != type_)
                throw std::invalid_argument("dense transformer dtype mismatch");
            if (linear->is_quantized())
                quant_input_channels_ = std::max(quant_input_channels_, linear->input_channels());
        }
        for (const auto &n : {block.attention_norm, block.feedforward_norm})
            if (!n.weight.data || !n.bias.data || n.weight.type != type_ || n.bias.type != type_ ||
                n.weight.numel() != size_t(hidden_) || n.bias.numel() != size_t(hidden_) ||
                !(n.epsilon > 0))
                throw std::invalid_argument("dense transformer norm mismatch");
    }
}
TransformerEncoder::Workspace::Workspace(const TransformerEncoder &encoder, int batch, int time, int rows_capacity)
    : max_batch(batch), max_time(time), max_rows(rows_capacity) {
    if (batch < 1 || time < 1 || rows_capacity < 0 ||
        int64_t(batch) * time > INT32_MAX)
        throw std::invalid_argument("invalid encoder workspace capacity");
    if (!max_rows) max_rows = batch * time;
    size_t rows = size_t(max_rows);
    auto storage = [&](int width) { return Buffer(rows * width * bytes(encoder.type_)); };
    projected = storage(3 * encoder.hidden_);
    attended = storage(encoder.hidden_);
    update = storage(encoder.hidden_);
    expanded = storage(encoder.blocks_.front().up.output_channels());
    lse = Buffer(rows * encoder.heads_ * sizeof(float));
    linear = Linear::Workspace(max_rows, encoder.quant_input_channels_);
}
void TransformerEncoder::prepare(Workspace &workspace, int max_rows) const {
    if (!max_rows) max_rows = workspace.max_rows;
    auto hidden_states = workspace.update.view(1, hidden_, type_);
    auto attention_states = workspace.attended.view(1, hidden_, type_);
    auto projected_states = workspace.projected.view(1, 3 * hidden_, type_);
    for (const auto &block : blocks_) {
        auto expanded_states = workspace.expanded.view(1, block.up.output_channels(), type_);
        block.qkv.prepare_range(hidden_states, projected_states, max_rows);
        block.out.prepare_range(attention_states, hidden_states, max_rows);
        block.up.prepare_range(hidden_states, expanded_states, max_rows, block.activation);
        block.down.prepare_range(expanded_states, hidden_states, max_rows);
    }
}
TensorView TransformerEncoder::forward(TensorView hidden_states, ForwardContext &context) const {
    Tensor::Scope scratch;
    const bool preparing = context.mode == ForwardContext::Prepare;
    Workspace workspace(*this, preparing ? 1 : context.max_batch,
                         preparing ? 1 : context.max_time, preparing ? 1 : context.max_rows);
    if (preparing && workspace.projected.data())
        prepare(workspace, context.max_rows);
    else if (context.mode == ForwardContext::Execute)
        forward(workspace, hidden_states, context.batch);
    return context.batch.output.numel() ? context.batch.output : hidden_states;
}
void TransformerEncoder::forward(Workspace &workspace, TensorView hidden_states,
                                 const ForwardBatch &attention) const {
    const int layer_count = int(blocks_.size());
    int batch = attention.batch(), time = attention.time(), prefix = attention.prefix;
    int rows = attention.rows();
    bool cached = attention.cache.layout.valid_paged();
    if (batch < 1 || batch > workspace.max_batch || time < 1 || time > workspace.max_time || prefix < 0 ||
        prefix > time || rows < 1 ||
        rows > workspace.max_rows || hidden_states.rows != rows || hidden_states.cols != hidden_ ||
        hidden_states.type != type_ || (cached && attention.capacity < time) ||
        (attention.output.data &&
         (!attention.output_rows || attention.output.rows < 1 ||
          attention.output.cols != hidden_ || attention.output.type != type_)) ||
        (attention.is_decode() && (!cached || time != 1 || attention.sequence.addressing())))
        throw std::invalid_argument("dense encoder execution bounds");
    auto projected = workspace.projected.view(rows, 3 * hidden_, type_);
    auto attended = workspace.attended.view(rows, hidden_, type_);
    auto update = workspace.update.view(rows, hidden_, type_);
    bool projected_ready = false, input_ready = false;
    for (int layer_index = 0; layer_index < layer_count; ++layer_index) {
        const auto &block = blocks_[layer_index];
        auto query = projected.columns(0, hidden_), key = projected.columns(hidden_, hidden_),
             value = projected.columns(2 * hidden_, hidden_);
        CacheLayout cache = attention.cache.layout;
        cache.layer = layer_index;
        if (cached) {
            key = attention.cache.keys;
            value = attention.cache.values;
        }
        bool qkv_fused = projected_ready;
        projected_ready = false;
        if (!qkv_fused)
            qkv_fused = attention.is_decode() && !block.qkv.is_quantized() &&
                ops::matmul_qkv(hidden_states, block.qkv.dense_weight(),
                               {query, key, value, block.qkv.bias(), cache, attention.positions,
                                hidden_, attention.capacity});
        if (!qkv_fused) {
            if (input_ready)
                block.qkv.project(block.qkv.quantized_view(workspace.linear, rows), projected);
            else
                block.qkv.project(workspace.linear, hidden_states, projected);
            if (cached) {
                ops::write_qkv(projected, query, key, value, attention.positions, batch, time, 0,
                        attention.capacity, heads_, heads_, hidden_ / heads_, {}, {}, 0,
                        false, {}, 0, attention.sequence.addressing(), cache, block.qkv.bias());
            } else if (block.qkv.bias().data) {
                ops::pointwise( projected, projected, {block.qkv.bias()});
            }
        }
        input_ready = false;
        auto attend = [&](TensorView query, TensorView output, int query_time, int key_time,
                          bool causal, DenseAttentionLayout layout = {}) {
            if (cached)
                prefill_attention( query, key, value, output, workspace.lse.ptr<float>(), batch,
                              query_time, 0, attention.capacity, heads_, heads_, hidden_ / heads_,
                              causal, 0, nullptr, cache, key_time, 0, layout);
            else
            {
                if (!layout.sequences && attention.sequence.cu_seqlens) {
                    layout.cu_seqlens_q = attention.sequence.cu_seqlens;
                    layout.cu_seqlens_k = attention.sequence.cu_seqlens;
                }
                dense_attention( query, key, value, output, workspace.lse.ptr<float>(), batch, query_time,
                            key_time, heads_, hidden_ / heads_, causal, 0, layout);
            }
        };
        if (attention.is_decode()) {
            decode_attention( query, key, value, attended, workspace.lse.ptr<float>(), attention.positions,
                         batch, heads_, heads_, cache, {},
                         attention.decode_partitions);
        } else if (attention.bidirectional || attention.causal) {
            if (attention.bidirectional)
                attend(query, attended, time, time, false, {0, 0, attention.bidirectional});
            if (attention.causal)
                attend(query, attended, time, time, true, {0, 0, attention.causal});
        } else if (!prefix) {
            attend(query, attended, time, time, false);
        } else {
            DenseAttentionLayout layout{time, time};
            attend(query, attended, prefix, prefix, false, layout);
            if (prefix < time)
                attend(query.slice(prefix, rows - prefix), attended.slice(prefix, rows - prefix),
                       time - prefix, time, true, layout);
        }
        block.out.project(workspace.linear, attended, update);
        ops::NormFusion f;
        f.residual = hidden_states;
        f.projection_bias = block.out.bias();
        auto expanded = workspace.expanded.view(rows, block.up.output_channels(), type_);
        ops::Epilogue encoder;
        encoder.activation = block.activation;
        auto down_input = block.down.is_quantized() ? block.down.quantized_view(workspace.linear, rows)
                                                : RowScaledView{};
        auto residual = hidden_states;
        bool up_fused = block.up.normalized_forward(workspace.linear, update, attended, expanded,
                                                 block.attention_norm.weight, block.attention_norm.bias,
                                                 block.attention_norm.epsilon, f, encoder, nullptr, down_input);
        if (up_fused) {
            // Attention has consumed this buffer. Retain the normalized skip
            // here until the feedforward norm writes back to x.
            residual = attended;
        } else {
            ops::layer_norm( update, hidden_states, block.attention_norm.weight, block.attention_norm.bias,
                            block.attention_norm.epsilon, f);
            block.up.forward(workspace.linear, hidden_states, expanded, encoder, down_input);
        }
        if (block.down.is_quantized())
            block.down.project(down_input, update);
        else
            block.down.project(workspace.linear, expanded, update);
        f.residual = residual;
        f.projection_bias = block.down.bias();
        bool selected = layer_index + 1 == layer_count && attention.output.data;
        f.source_rows = selected ? attention.output_rows : nullptr;
        if (attention.is_decode() && up_fused && layer_index + 1 < layer_count) {
            auto next_cache = attention.cache.layout;
            next_cache.layer = layer_index + 1;
            QkvProjectionOutput next{query, key, value, blocks_[layer_index + 1].qkv.bias(), next_cache,
                                     attention.positions, hidden_, attention.capacity};
            projected_ready = blocks_[layer_index + 1].qkv.normalized_forward(update, hidden_states, query, block.feedforward_norm.weight, block.feedforward_norm.bias,
                block.feedforward_norm.epsilon, f, {}, &next);
        }
        if (!projected_ready) {
            if (layer_index + 1 < layer_count && blocks_[layer_index + 1].qkv.is_quantized()) {
                blocks_[layer_index + 1].qkv.normalize_input(workspace.linear, update,
                                                    block.feedforward_norm.weight,
                                                    block.feedforward_norm.bias,
                                                    block.feedforward_norm.epsilon, f, hidden_states);
                input_ready = true;
            } else {
                ops::layer_norm( update, selected ? attention.output : hidden_states,
                                block.feedforward_norm.weight, block.feedforward_norm.bias,
                                block.feedforward_norm.epsilon, f);
            }
        }
    }
}
} // namespace tts
