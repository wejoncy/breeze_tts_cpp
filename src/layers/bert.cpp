#include "ops/basic.h"
#include "layers/bert.h"
#include <cmath>

namespace tts {
namespace {
AffineNorm norm(Weights &weights, const std::string &name, float eps) {
    return {weights.get(name + ".weight"), weights.get(name + ".bias"), eps};
}
std::vector<EncoderBlock> bert_blocks(Weights &weights, const Json &configuration, const std::string &prefix, int count,
                                     bool quantize) {
    std::vector<EncoderBlock> result;
    if (count < 1 || count > configuration.at("num_hidden_layers").get<int>())
        throw std::invalid_argument("BERT output layer outside configured stack");
    int hidden = configuration.at("hidden_size");
    Type type = weights.entry(prefix + ".embeddings.word_embeddings.weight").type;
    for (int layer_index = 0; layer_index < count; ++layer_index) {
        auto layer_prefix = prefix + ".encoder.layer." + std::to_string(layer_index);
        for (const char *projection : {"query", "key", "value"}) {
            auto name = layer_prefix + ".attention.self." + projection;
            weights.validate(name + ".weight", {hidden, hidden}, type);
            weights.validate(name + ".bias", {hidden}, type);
        }
        if (configuration.contains("intermediate_size")) {
            int intermediate = configuration.at("intermediate_size");
            weights.validate(layer_prefix + ".intermediate.dense.weight", {intermediate, hidden}, type);
            weights.validate(layer_prefix + ".output.dense.weight", {hidden, intermediate}, type);
        }
        auto bias = weights.fused(layer_prefix + ".qkv.bias",
                            {layer_prefix + ".attention.self.query.bias", layer_prefix + ".attention.self.key.bias",
                             layer_prefix + ".attention.self.value.bias"});
        auto qkv = Linear::fused(weights, layer_prefix + ".qkv.weight",
                                {layer_prefix + ".attention.self.query.weight", layer_prefix + ".attention.self.key.weight",
                                 layer_prefix + ".attention.self.value.weight"}, bias, quantize);
        result.push_back({std::move(qkv), Linear(weights, layer_prefix + ".attention.output.dense"),
                          Linear(weights, layer_prefix + ".intermediate.dense", quantize),
                          Linear(weights, layer_prefix + ".output.dense", quantize),
                          norm(weights, layer_prefix + ".attention.output.LayerNorm", configuration.at("layer_norm_eps")),
                          norm(weights, layer_prefix + ".output.LayerNorm", configuration.at("layer_norm_eps")),
                          ops::Activation::Gelu});
    }
    return result;
}
} // namespace
Bert::Bert(Weights &weights, const Json &cfg, const std::string &prefix, int output_layers, bool quantize)
    : words_(weights.get(prefix + ".embeddings.word_embeddings.weight")),
      positions_(weights.get(prefix + ".embeddings.position_embeddings.weight")),
      segments_(weights.get(prefix + ".embeddings.token_type_embeddings.weight")),
      embedding_norm_(norm(weights, prefix + ".embeddings.LayerNorm", cfg.at("layer_norm_eps"))),
      encoder_(bert_blocks(weights, cfg, prefix, output_layers, quantize), cfg.at("num_attention_heads")) {
    if (cfg.at("position_embedding_type") != "absolute" || cfg.at("hidden_act") != "gelu")
        throw std::invalid_argument("unsupported BERT position/activation configuration");
    int hidden = cfg.at("hidden_size");
    weights.validate(prefix + ".embeddings.word_embeddings.weight",
               {cfg.contains("vocab_size") ? cfg.at("vocab_size").get<int>() : words_.rows, hidden},
               encoder_.type());
    weights.validate(prefix + ".embeddings.position_embeddings.weight",
               {cfg.at("max_position_embeddings"), hidden}, encoder_.type());
    weights.validate(
        prefix + ".embeddings.token_type_embeddings.weight",
        {cfg.contains("type_vocab_size") ? cfg.at("type_vocab_size").get<int>() : segments_.rows,
         hidden},
        encoder_.type());
    if (encoder_.hidden() != hidden || embedding_norm_.weight.type != encoder_.type() ||
        embedding_norm_.bias.type != encoder_.type() ||
        embedding_norm_.weight.numel() != size_t(hidden) ||
        embedding_norm_.bias.numel() != size_t(hidden) || !(embedding_norm_.epsilon > 0) ||
        !std::isfinite(embedding_norm_.epsilon))
        throw std::invalid_argument("BERT embedding norm/configuration mismatch: " + prefix);
}
TensorView Bert::forward(Workspace &workspace, const int *ids,
                         const ForwardBatch &attention) const {

    int batch = attention.batch(), time = attention.time();
    if (batch < 1 || batch > workspace.encoder.max_batch || time < 1 || time > workspace.encoder.max_time)
        throw std::invalid_argument("BERT exceeds initialized workspace");
    int rows = attention.rows();
    if (attention.sequence.addressing() && !attention.positions)
        throw std::invalid_argument("packed BERT requires token positions");
    auto hidden_states = workspace.features.view(rows, encoder_.hidden(), encoder_.type());
    ops::EmbeddingFusion embedding;
    embedding.positions = positions_;
    embedding.segment = segments_;
    embedding.norm_weight = embedding_norm_.weight;
    embedding.norm_bias = embedding_norm_.bias;
    embedding.epsilon = embedding_norm_.epsilon;
    embedding.sequence_length = time;
    embedding.position_ids = attention.positions;
    ops::embedding( words_, ids, hidden_states, embedding);
    encoder_.forward(workspace.encoder, hidden_states, attention);
    return hidden_states;
}
TensorView Bert::forward(TensorView input_ids, ForwardContext &context) const {
    if (context.mode == ForwardContext::Prepare) {
        Workspace workspace(*this, 1, 1, &context.storage, 1);
        if (workspace.features.data()) encoder_.prepare(workspace.encoder, context.max_rows);
        return workspace.features.view(1, encoder_.hidden(), encoder_.type());
    }
    Workspace workspace(*this, context.max_batch, context.max_time, &context.storage,
                        context.max_rows);
    if (context.mode == ForwardContext::Execute)
        forward(workspace, input_ids.ptr<int>(), context.batch);
    return context.batch.output.numel() ? context.batch.output
        : workspace.features.view(context.batch.rows(), encoder_.hidden(), encoder_.type());
}
} // namespace tts
