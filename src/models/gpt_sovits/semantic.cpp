#include "ops/basic.h"
#include "models/gpt_sovits/semantic.h"
#include <cmath>

namespace tts {
namespace {
std::vector<EncoderBlock> semantic_blocks(Weights &weights, const std::string &prefix, const Json &configuration) {
    std::vector<EncoderBlock> result;
    const bool quantize = (weights.quantization() != Quantization::None);
    for (int layer_index = 0; layer_index < configuration.at("n_layer").get<int>(); ++layer_index) {
        auto layer_prefix = prefix + ".h.layers." + std::to_string(layer_index);
        auto norm = [&](const char *name) -> AffineNorm {
            return {weights.get(layer_prefix + name + ".weight"), weights.get(layer_prefix + name + ".bias"), 1.e-5f};
        };
        result.push_back(
            {Linear::fused(weights, layer_prefix + ".qkv.weight", {layer_prefix + ".self_attn.in_proj_weight"},
                           weights.get(layer_prefix + ".self_attn.in_proj_bias"), quantize),
             Linear(weights, layer_prefix + ".self_attn.out_proj"), Linear(weights, layer_prefix + ".linear1", quantize),
             Linear(weights, layer_prefix + ".linear2", quantize), norm(".norm1"), norm(".norm2"),
             ops::Activation::Relu});
    }
    return result;
}
} // namespace
Semantic::Semantic(Weights &weights, const std::string &prefix, const Json &configuration, int context)
    : phones_(weights.get(prefix + ".ar_text_embedding.word_embeddings.weight")),
      audio_(weights.get(prefix + ".ar_audio_embedding.word_embeddings.weight")),
      phone_alpha_(weights.get(prefix + ".ar_text_position.alpha")),
      audio_alpha_(weights.get(prefix + ".ar_audio_position.alpha")),
      bert_(weights, prefix + ".bert_proj"), head_(weights.get(prefix + ".ar_predict_layer.weight")),
      transformer_(semantic_blocks(weights, prefix, configuration), configuration.at("head")) {
    int hidden = transformer_.hidden();
    weights.validate(prefix + ".ar_text_embedding.word_embeddings.weight",
               {configuration.at("phoneme_vocab_size"), hidden}, type());
    weights.validate(prefix + ".ar_audio_embedding.word_embeddings.weight",
               {configuration.at("vocab_size"), hidden}, type());
    if (context < 1 || configuration.at("embedding_dim") != hidden || configuration.at("hidden_dim") != hidden ||
        phone_alpha_.type != type() || phone_alpha_.numel() != 1 || audio_alpha_.type != type() ||
        audio_alpha_.numel() != 1 || bert_.output_channels() != hidden ||
        head_.input_channels() != hidden || head_.output_channels() != configuration.at("vocab_size"))
        throw std::invalid_argument("semantic embedding/context mismatch");
    int dim = transformer_.hidden();
    std::vector<float> positions(size_t(context) * dim);
    for (int layer_prefix = 0; layer_prefix < context; ++layer_prefix)
        for (int dimension_index = 0; dimension_index < dim / 2; ++dimension_index) {
            float angle = layer_prefix * std::exp(-std::log(10000.f) * (2 * dimension_index) / dim);
            positions[size_t(layer_prefix) * dim + 2 * dimension_index] = std::sin(angle);
            positions[size_t(layer_prefix) * dim + 2 * dimension_index + 1] = std::cos(angle);
        }
    positions_ =
        weights.generated(prefix + ".position_table", positions, context, dim, transformer_.type());
}
TensorView Semantic::forward(TensorInputs inputs, ForwardContext &context) const {
    if (context.mode == ForwardContext::Prepare) {
        auto hidden_states = context.tensor({1, hidden()}, type());
        auto logits = context.tensor({1, head_.output_channels()}, type());
        TransformerEncoder::Workspace workspace(transformer_, 1, 1, 1);
        if (hidden_states.data) {
            transformer_.prepare(workspace, context.max_rows);
            head_.prepare_range(hidden_states, logits, context.max_batch);
        }
        return logits;
    }
    auto hidden_states = context.tensor({context.max_rows, hidden()}, type())
                             .slice(0, context.batch.rows());
    auto logits = context.tensor({context.max_batch, head_.output_channels()}, type())
                      .slice(0, context.batch.batch());
    TransformerEncoder::Workspace workspace(transformer_, context.max_batch, context.max_time, context.max_rows);
    auto attention = context.batch;
    auto final_states = hidden_states;
    if (!attention.is_decode()) {
        final_states = context.tensor({context.max_batch, hidden()}, type())
                           .slice(0, attention.batch());
        attention.output = final_states;
    }
    if (context.mode == ForwardContext::Execute) {

        ops::EmbeddingFusion embedding;
        embedding.positions = positions_;
        if (attention.is_decode()) {
            embedding.position_alpha = audio_alpha_;
            embedding.position_ids = inputs["audio_positions"].ptr<int>();
            embedding.id_rows = inputs["previous_codes"].ptr<const int *>();
            ops::embedding( audio_, nullptr, hidden_states, embedding);
        } else {
            embedding.position_alpha = phone_alpha_;
            embedding.residual = inputs["encoded_features"];
            embedding.residual_default = bert_.bias();
            embedding.output_rows = inputs["phone_output_rows"].ptr<int>();
            embedding.position_ids = inputs["phone_positions"].ptr<int>();
            embedding.residual_rows = inputs["phone_feature_rows"].ptr<int>();
            embedding.residual_sources = inputs["reference_features"].ptr<const void *>();
            embedding.input_rows = inputs["phones"].rows;
            ops::embedding( phones_, inputs["phones"].ptr<int>(), hidden_states,
                           embedding);
            embedding = {};
            embedding.positions = positions_;
            embedding.position_alpha = audio_alpha_;
            embedding.output_rows = inputs["code_output_rows"].ptr<int>();
            embedding.position_ids = inputs["code_positions"].ptr<int>();
            embedding.input_rows = inputs["codes"].rows;
            ops::embedding( audio_, inputs["codes"].ptr<int>(), hidden_states,
                           embedding);
        }
        transformer_.forward(workspace, hidden_states, attention);
        head_.forward(final_states, logits);
    }
    return logits;
}
} // namespace tts
