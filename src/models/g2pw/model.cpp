#include "ops/basic.h"
#include "models/g2pw/model.h"
#include "io/weights.h"
#include "ops/pronunciation.h"
#include <cstdint>

namespace tts {
TensorView Pronunciation::forward(TensorInputs inputs, ForwardContext &context) const {
    const auto positions = inputs["positions"], characters = inputs["characters"];
    const int queries = positions.rows;
    if (positions.type != Type::I32 || characters.type != Type::I32 || positions.cols != 1 ||
        characters.cols != 1 || characters.rows != queries || queries < 1 || queries > context.max_rows)
        throw std::invalid_argument("pronunciation query shape/type mismatch");
    auto selected = context.tensor({queries, bert_.hidden()}, bert_.type());
    {
        Tensor::Scope encoder_scope;
        auto encoder_context = context;
        encoder_context.batch.output = selected;
        encoder_context.batch.output_rows = positions.ptr<int>();
        encoder_context.batch.positions = context.batch.sequence.row_position;
        bert_.forward(inputs["input_ids"], encoder_context);
    }
    auto logits = context.tensor({queries, classifier_.output_channels()}, Type::F32);
    auto pos_logits = context.tensor({queries, pos_classifier_.output_channels()}, Type::F32);
    auto labels = context.tensor({queries, 1}, Type::I32);
    if (context.mode == ForwardContext::Prepare && selected.data) {
        classifier_.prepare_range(selected.slice(0), logits.slice(0), context.max_rows);
        pos_classifier_.prepare_range(selected.slice(0), pos_logits.slice(0), context.max_rows);
    } else if (context.mode == ForwardContext::Execute) {
        classifier_.project(selected, logits);
        pos_classifier_.project(selected, pos_logits);
        ops::select_pronunciation(logits, pos_logits, classifier_.bias(), pos_classifier_.bias(),
            descriptor_bias_, char_descriptor_, second_order_descriptor_, legal_, characters.ptr<int>(), labels.ptr<int>());
    }
    return labels;
}
Pronunciation::Pronunciation(Weights &w, const std::string &prefix, const Json &metadata)
    : bert_(w, metadata.at("model"), prefix + ".bert",
            metadata.at("model").at("num_hidden_layers"), (w.quantization() != Quantization::None)),
      classifier_(w, prefix + ".classifier"), pos_classifier_(w, prefix + ".pos_classifier"),
      descriptor_bias_(w.get(prefix + ".descriptor_bias.weight")),
      char_descriptor_(w.get(prefix + ".char_descriptor.weight")),
      second_order_descriptor_(w.get(prefix + ".second_order_descriptor.weight")) {
    int labels = classifier_.output_channels(), chars = char_descriptor_.rows;
    const int max_tokens = metadata.at("max_tokens");
    if (max_tokens < 1 || max_tokens > metadata.at("model").at("max_position_embeddings") ||
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
} // namespace tts
