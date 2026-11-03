#include "models/gpt_sovits/semantic.h"
#include <cmath>

namespace tts::gpt_sovits {
namespace {
std::vector<EncoderBlock> semantic_blocks(Weights &w, const Json &cfg) {
    std::vector<EncoderBlock> result;
    const bool quantize = w.backend().has_quantization();
    for (int i = 0; i < cfg.at("n_layer").get<int>(); ++i) {
        auto p = "semantic.model.h.layers." + std::to_string(i);
        auto norm = [&](const char *name) -> AffineNorm {
            return {w.get(p + name + ".weight"), w.get(p + name + ".bias"), 1.e-5f};
        };
        result.push_back(
            {Linear::fused(w, p + ".qkv.weight", {p + ".self_attn.in_proj_weight"},
                           w.get(p + ".self_attn.in_proj_bias"), quantize),
             Linear(w, p + ".self_attn.out_proj"), Linear(w, p + ".linear1", quantize),
             Linear(w, p + ".linear2", quantize), norm(".norm1"), norm(".norm2"),
             ops::Activation::Relu});
    }
    return result;
}
} // namespace
Semantic::Semantic(Weights &w, const Json &cfg, int context)
    : phones_(w.get("semantic.model.ar_text_embedding.word_embeddings.weight")),
      audio_(w.get("semantic.model.ar_audio_embedding.word_embeddings.weight")),
      phone_alpha_(w.get("semantic.model.ar_text_position.alpha")),
      audio_alpha_(w.get("semantic.model.ar_audio_position.alpha")),
      bert_(w, "semantic.model.bert_proj"), head_(w.get("semantic.model.ar_predict_layer.weight")),
      transformer_(semantic_blocks(w, cfg), cfg.at("head")) {
    int hidden = transformer_.hidden();
    w.validate("semantic.model.ar_text_embedding.word_embeddings.weight",
               {cfg.at("phoneme_vocab_size"), hidden}, type());
    w.validate("semantic.model.ar_audio_embedding.word_embeddings.weight",
               {cfg.at("vocab_size"), hidden}, type());
    if (context < 1 || cfg.at("embedding_dim") != hidden || cfg.at("hidden_dim") != hidden ||
        phone_alpha_.type != type() || phone_alpha_.numel() != 1 || audio_alpha_.type != type() ||
        audio_alpha_.numel() != 1 || bert_.output_channels() != hidden ||
        head_.input_channels() != hidden || head_.output_channels() != cfg.at("vocab_size"))
        throw std::invalid_argument("semantic embedding/context mismatch");
    int dim = transformer_.hidden();
    std::vector<float> positions(size_t(context) * dim);
    for (int p = 0; p < context; ++p)
        for (int d = 0; d < dim / 2; ++d) {
            float angle = p * std::exp(-std::log(10000.f) * (2 * d) / dim);
            positions[size_t(p) * dim + 2 * d] = std::sin(angle);
            positions[size_t(p) * dim + 2 * d + 1] = std::cos(angle);
        }
    positions_ =
        w.generated("semantic.position_table", positions, context, dim, transformer_.type());
}
TensorView Semantic::encode_features(Workspace &w, TensorView features) const {
    if (features.rows < 1 || features.cols != bert_.input_channels() || features.type != type() ||
        size_t(features.rows) * hidden() * bytes(type()) > w.bert_projection.size())
        throw std::invalid_argument("semantic conditioning exceeds initialized workspace");
    auto projection = w.bert_projection.view(features.rows, hidden(), type());
    bert_.forward(features, projection);
    return projection;
}
void Semantic::prefill(Workspace &w, const PrefillBatch &p, TensorView logits) const {
    auto &gpu = backend_for(logits);
    const auto &context = p.context;
    int batch = context.batch(), time = context.time(), rows = context.rows();
    if (batch > w.transformer.max_batch || time > w.transformer.max_time ||
        rows > int64_t(w.transformer.max_batch) * w.transformer.max_time ||
        (p.encoded_features.data && (p.encoded_features.rows < 1 ||
         p.encoded_features.cols != hidden() || p.encoded_features.type != type())) || logits.rows != batch ||
        logits.cols != head_.output_channels() ||
        !context.sequence.cu_seqlens || !p.phone_rows.output_rows ||
        !p.audio_rows.output_rows || !context.output_rows)
        throw std::invalid_argument("invalid semantic prefill batch");
    auto x = w.hidden.view(rows, hidden(), type());
    auto f = p.phone_rows;
    f.positions = positions_;
    f.position_alpha = phone_alpha_;
    f.residual = p.encoded_features;
    f.residual_default = bert_.bias();
    ops::embedding(gpu.stream, phones_, p.phones, x, f);
    f = p.audio_rows;
    f.positions = positions_;
    f.position_alpha = audio_alpha_;
    ops::embedding(gpu.stream, audio_, p.codes, x, f);
    auto attention = context;
    auto last = w.last.view(batch, hidden(), type());
    attention.output = last;
    transformer_.forward(w.transformer, x, attention);
    head_.forward(last, logits);
}
void Semantic::decode(Workspace &w, const int *codes, const int *audio_positions,
                      const ForwardBatch &input, TensorView logits,
                      const int *const *code_rows) const {
    auto &gpu = backend_for(logits);
    int batch = input.batch();
    if (batch < 1 || batch > w.transformer.max_batch || (!codes && !code_rows) ||
        !audio_positions || !input.decode || input.time() != 1 || input.sequence.addressing() ||
        (!input.positions && !input.cache.layout.sequences) ||
        !input.cache.layout.valid_paged() || logits.rows != batch ||
        logits.cols != head_.output_channels())
        throw std::invalid_argument("invalid semantic decode geometry/cache");
    auto x = w.hidden.view(batch, hidden(), type());
    ops::EmbeddingFusion f;
    f.positions = positions_;
    f.position_alpha = audio_alpha_;
    f.position_ids = audio_positions;
    f.id_rows = code_rows;
    ops::embedding(gpu.stream, audio_, codes, x, f);
    transformer_.forward(w.transformer, x, input);
    head_.forward(x, logits);
}
} // namespace tts::gpt_sovits
