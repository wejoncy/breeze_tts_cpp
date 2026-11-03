#include "layers/bert.h"
#include <cmath>

namespace tts {
namespace {
AffineNorm norm(Weights &w, const std::string &name, float eps) {
    return {w.get(name + ".weight"), w.get(name + ".bias"), eps};
}
std::vector<EncoderBlock> bert_blocks(Weights &w, const Json &c, const std::string &prefix, int count,
                                     bool quantize) {
    std::vector<EncoderBlock> result;
    if (count < 1 || count > c.at("num_hidden_layers").get<int>())
        throw std::invalid_argument("BERT output layer outside configured stack");
    int hidden = c.at("hidden_size");
    Type type = w.entry(prefix + ".embeddings.word_embeddings.weight").type;
    for (int i = 0; i < count; ++i) {
        auto p = prefix + ".encoder.layer." + std::to_string(i);
        for (const char *projection : {"query", "key", "value"}) {
            auto name = p + ".attention.self." + projection;
            w.validate(name + ".weight", {hidden, hidden}, type);
            w.validate(name + ".bias", {hidden}, type);
        }
        if (c.contains("intermediate_size")) {
            int intermediate = c.at("intermediate_size");
            w.validate(p + ".intermediate.dense.weight", {intermediate, hidden}, type);
            w.validate(p + ".output.dense.weight", {hidden, intermediate}, type);
        }
        auto bias = w.fused(p + ".qkv.bias",
                            {p + ".attention.self.query.bias", p + ".attention.self.key.bias",
                             p + ".attention.self.value.bias"});
        auto qkv = Linear::fused(w, p + ".qkv.weight",
                                {p + ".attention.self.query.weight", p + ".attention.self.key.weight",
                                 p + ".attention.self.value.weight"}, bias, quantize);
        result.push_back({std::move(qkv), Linear(w, p + ".attention.output.dense"),
                          Linear(w, p + ".intermediate.dense", quantize),
                          Linear(w, p + ".output.dense", quantize),
                          norm(w, p + ".attention.output.LayerNorm", c.at("layer_norm_eps")),
                          norm(w, p + ".output.LayerNorm", c.at("layer_norm_eps")),
                          ops::Activation::Gelu});
    }
    return result;
}
} // namespace
Bert::Bert(Weights &w, const Json &cfg, const std::string &prefix, int output_layers, bool quantize)
    : words_(w.get(prefix + ".embeddings.word_embeddings.weight")),
      positions_(w.get(prefix + ".embeddings.position_embeddings.weight")),
      segments_(w.get(prefix + ".embeddings.token_type_embeddings.weight")),
      embedding_norm_(norm(w, prefix + ".embeddings.LayerNorm", cfg.at("layer_norm_eps"))),
      encoder_(bert_blocks(w, cfg, prefix, output_layers, quantize), cfg.at("num_attention_heads")) {
    if (cfg.at("position_embedding_type") != "absolute" || cfg.at("hidden_act") != "gelu")
        throw std::invalid_argument("unsupported BERT position/activation configuration");
    int hidden = cfg.at("hidden_size");
    w.validate(prefix + ".embeddings.word_embeddings.weight",
               {cfg.contains("vocab_size") ? cfg.at("vocab_size").get<int>() : words_.rows, hidden},
               encoder_.type());
    w.validate(prefix + ".embeddings.position_embeddings.weight",
               {cfg.at("max_position_embeddings"), hidden}, encoder_.type());
    w.validate(
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
TensorView Bert::forward(Workspace &w, const int *ids,
                         const ForwardBatch &attention) const {
    auto &gpu = backend_for(w.features.device());
    int batch = attention.batch(), time = attention.time();
    if (batch < 1 || batch > w.encoder.max_batch || time < 1 || time > w.encoder.max_time)
        throw std::invalid_argument("BERT exceeds initialized workspace");
    int rows = attention.rows();
    if (attention.sequence.addressing() && !attention.positions)
        throw std::invalid_argument("packed BERT requires token positions");
    auto x = w.features.view(rows, encoder_.hidden(), encoder_.type());
    ops::EmbeddingFusion f;
    f.positions = positions_;
    f.segment = segments_;
    f.norm_weight = embedding_norm_.weight;
    f.norm_bias = embedding_norm_.bias;
    f.epsilon = embedding_norm_.epsilon;
    f.sequence_length = time;
    f.position_ids = attention.positions;
    ops::embedding(gpu.stream, words_, ids, x, f);
    encoder_.forward(w.encoder, x, attention);
    return x;
}
} // namespace tts
