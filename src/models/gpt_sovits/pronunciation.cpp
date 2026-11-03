#include "models/gpt_sovits/pronunciation.h"
#include "ops/pronunciation.h"
#include <cstdint>

namespace tts::gpt_sovits {
Pronunciation::Pronunciation(Weights &w, const Json &metadata)
    : bert_(w, metadata.at("model"), "g2pw.bert",
            metadata.at("model").at("num_hidden_layers"), w.backend().has_quantization()),
      classifier_(w, "g2pw.classifier"), pos_classifier_(w, "g2pw.pos_classifier"),
      descriptor_bias_(w.get("g2pw.descriptor_bias.weight")),
      char_descriptor_(w.get("g2pw.char_descriptor.weight")),
      second_order_descriptor_(w.get("g2pw.second_order_descriptor.weight")),
      max_tokens_(metadata.at("max_tokens")) {
    int labels = classifier_.output_channels(), chars = char_descriptor_.rows;
    if (max_tokens_ < 1 || max_tokens_ > metadata.at("model").at("max_position_embeddings") ||
        bert_.hidden() != metadata.at("model").at("hidden_size") ||
        classifier_.input_channels() != bert_.hidden() ||
        pos_classifier_.input_channels() != bert_.hidden() ||
        classifier_.activation_type() != bert_.type() ||
        pos_classifier_.activation_type() != bert_.type() ||
        !classifier_.bias().data || !pos_classifier_.bias().data ||
        labels != int(metadata.at("labels").size()) ||
        chars != int(metadata.at("characters").size()) || labels < 1 || chars < 1 ||
        descriptor_bias_.type != Type::F32 || char_descriptor_.type != Type::F32 ||
        second_order_descriptor_.type != Type::F32 || descriptor_bias_.rows != 1 ||
        descriptor_bias_.cols != labels || char_descriptor_.cols != labels ||
        second_order_descriptor_.cols != labels ||
        int64_t(second_order_descriptor_.rows) != int64_t(chars) * pos_classifier_.output_channels())
        throw std::invalid_argument("G2PW configuration/parameter dimensions disagree");
    int words = (labels - 1) / 32 + 1;
    std::vector<uint32_t> legal(size_t(chars) * words, 0);
    const auto &allowed = metadata.at("allowed");
    if (allowed.size() != size_t(chars))
        throw std::invalid_argument("G2PW pronunciation mask count");
    for (int ch = 0; ch < chars; ++ch) {
        if (!allowed[ch].is_array() || allowed[ch].empty())
            throw std::invalid_argument("G2PW character has no legal pronunciation");
        for (const auto &value : allowed[ch]) {
            int label = value.get<int>();
            if (label < 0 || label >= labels)
                throw std::invalid_argument("G2PW pronunciation outside label vocabulary");
            legal[size_t(ch) * words + label / 32] |= uint32_t(1) << (label % 32);
        }
    }
    legal_ = w.custom("g2pw.legal_pronunciations", legal.data(), chars, words, Type::I32);
}
Pronunciation::Workspace::Workspace(const Pronunciation &p, int time, BufferAllocator *allocator)
    : bert(p.bert_, p.max_tokens_, time, allocator, time),
      selected(size_t(time) * p.bert_.hidden() * bytes(p.bert_.type()), allocator),
      logits(size_t(time) * p.classifier_.output_channels() * sizeof(float), allocator),
      pos_logits(size_t(time) * p.pos_classifier_.output_channels() * sizeof(float), allocator),
      max_time(time) {
    if (time > p.max_tokens_)
        throw std::invalid_argument("G2PW workspace exceeds configured token limit");
}
void Pronunciation::forward(Workspace &w, const int *ids, const int *positions,
                            const int *char_ids, const int *cu_seqlens, int time,
                            int *output_ids, int batch, const int *token_positions) const {
    auto &gpu = backend_for(w.selected.device());
    if (!ids || !positions || !char_ids || !cu_seqlens || !output_ids || !token_positions ||
        batch < 1 || batch > w.bert.encoder.max_batch || time < 1 || time > w.max_time)
        throw std::invalid_argument("G2PW execution exceeds initialized workspace");
    ForwardBatch attention(SequenceLayout{nullptr, nullptr, cu_seqlens, nullptr, time, batch, time});
    attention.output_rows = positions;
    attention.output = w.selected.view(time, bert_.hidden(), bert_.type());
    attention.positions = token_positions;
    bert_.forward(w.bert, ids, attention);
    auto logits = w.logits.view(time, classifier_.output_channels(), Type::F32);
    auto pos = w.pos_logits.view(time, pos_classifier_.output_channels(), Type::F32);
    classifier_.project(attention.output, logits);
    pos_classifier_.project(attention.output, pos);
    ops::select_pronunciation(gpu.stream, logits, pos, classifier_.bias(), pos_classifier_.bias(),
                             descriptor_bias_, char_descriptor_, second_order_descriptor_, legal_,
                             positions, char_ids, output_ids);
}
} // namespace tts::gpt_sovits
