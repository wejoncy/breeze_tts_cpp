#include "ops/basic.h"
#include "models/g2pw/model.h"
#include "ops/pronunciation.h"
#include <cstdint>

namespace tts {
TensorView Pronunciation::forward(TensorInputs inputs, ForwardContext &context) const {
    Workspace workspace(*this, context.max_rows);
    auto labels = context.tensor({context.max_rows, 1}, Type::I32);
    if (context.mode == ForwardContext::Prepare && workspace.selected.data()) {
        prepare();
        auto selected = workspace.selected.view(1, bert_.hidden(), bert_.type());
        classifier_.prepare_range(selected, workspace.logits.view(1, classifier_.output_channels(), Type::F32),
                                   context.max_rows);
        pos_classifier_.prepare_range(selected, workspace.pos_logits.view(1, pos_classifier_.output_channels(), Type::F32),
                                       context.max_rows);
    } else if (context.mode == ForwardContext::Execute) {
        forward(workspace, inputs["input_ids"].ptr<int>(), inputs["positions"].ptr<int>(),
            inputs["characters"].ptr<int>(), inputs["cu_seqlens"].ptr<int>(), inputs["input_ids"].rows,
            labels.ptr<int>(), context.max_batch, inputs["token_positions"].ptr<int>());
    }
    return labels.slice(0, int(context.parameters["queries"]));
}
Pronunciation::Pronunciation(Weights &w, const std::string &prefix, const Json &metadata)
    : bert_(w, metadata.at("model"), prefix + ".bert",
            metadata.at("model").at("num_hidden_layers"), (w.quantization() != Quantization::None)),
      classifier_(w, prefix + ".classifier"), pos_classifier_(w, prefix + ".pos_classifier"),
      descriptor_bias_(w.get(prefix + ".descriptor_bias.weight")),
      char_descriptor_(w.get(prefix + ".char_descriptor.weight")),
      second_order_descriptor_(w.get(prefix + ".second_order_descriptor.weight")),
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
    legal_ = w.custom(prefix + ".legal_pronunciations", legal.data(), chars, words, Type::I32);
}
Pronunciation::Workspace::Workspace(const Pronunciation &p, int time)
    : bert(p.bert_, p.max_tokens_, time, time),
      selected(size_t(time) * p.bert_.hidden() * bytes(p.bert_.type())),
      logits(size_t(time) * p.classifier_.output_channels() * sizeof(float)),
      pos_logits(size_t(time) * p.pos_classifier_.output_channels() * sizeof(float)),
      max_time(time) {
    if (time > p.max_tokens_)
        throw std::invalid_argument("G2PW workspace exceeds configured token limit");
}
void Pronunciation::forward(Workspace &w, const int *ids, const int *positions,
                            const int *char_ids, const int *cu_seqlens, int time,
                            int *output_ids, int batch, const int *token_positions) const {

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
    ops::select_pronunciation( logits, pos, classifier_.bias(), pos_classifier_.bias(),
                             descriptor_bias_, char_descriptor_, second_order_descriptor_, legal_,
                             positions, char_ids, output_ids);
}
} // namespace tts
