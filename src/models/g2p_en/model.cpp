#include "models/g2p_en/model.h"
#include <algorithm>

namespace tts {
namespace {
TensorView matrix(Weights &weights, const std::string &name) {
    const auto &entry = weights.entry("english." + name);
    if (entry.type != Type::F32 || entry.shape.empty() || entry.shape.size() > 2)
        throw std::invalid_argument("invalid grapheme model weight dtype/rank");
    return weights.get("english." + name);
}
GRUCell cell(Weights &weights, const std::string &name) {
    return {matrix(weights, name + "_w_ih"), matrix(weights, name + "_w_hh"),
            matrix(weights, name + "_b_ih"), matrix(weights, name + "_b_hh")};
}
int symbol(const std::vector<std::string> &alphabet, const char *name) {
    auto found = std::find(alphabet.begin(), alphabet.end(), name);
    if (found == alphabet.end()) throw std::invalid_argument("missing recurrent alphabet symbol");
    return int(found - alphabet.begin());
}
} // namespace
GraphemeToPhoneme::GraphemeToPhoneme(Weights &weights, const Json &metadata)
    : encoder_embedding_(matrix(weights, "enc_emb")), decoder_embedding_(matrix(weights, "dec_emb")),
      output_(matrix(weights, "fc_w")), output_bias_(matrix(weights, "fc_b")),
      encoder_(cell(weights, "enc")), decoder_(cell(weights, "dec")), max_steps_(metadata.at("decoder_steps")) {
    const auto graphemes = metadata.at("graphemes").get<std::vector<std::string>>();
    const auto phonemes = metadata.at("phonemes").get<std::vector<std::string>>();
    encoder_end_ = symbol(graphemes, "</s>");
    decoder_start_ = symbol(phonemes, "<s>"); decoder_end_ = symbol(phonemes, "</s>");
    if (!weights.device().is_cpu() || max_steps_ < 1 || encoder_.hidden() != decoder_.hidden() ||
        output_.cols != decoder_.hidden() || output_.rows != int(phonemes.size()) ||
        output_bias_.numel() != size_t(output_.rows) || !output_bias_.dense() ||
        encoder_embedding_.rows != int(graphemes.size()) || decoder_embedding_.rows != output_.rows ||
        encoder_embedding_.cols != encoder_.input_channels() || decoder_embedding_.cols != decoder_.input_channels())
        throw std::invalid_argument("recurrent alphabet/output geometry mismatch");
}
Tensor GraphemeToPhoneme::forward(TensorView ids, TensorView cu_seqlens, TensorView lengths) const {
    const int batch = cu_seqlens.rows - 1;
    if (batch < 1 || ids.device >= 0 || ids.type != Type::I32 || ids.cols != 1 || !ids.dense() ||
        (ids.numel() && !ids.data) || cu_seqlens.device >= 0 || cu_seqlens.type != Type::I32 ||
        cu_seqlens.cols != 1 || !cu_seqlens.dense() || !cu_seqlens.data || lengths.device >= 0 ||
        lengths.type != Type::I32 || lengths.rows != batch || lengths.cols != 1 || !lengths.data)
        throw std::invalid_argument("invalid recurrent input batch");
    const auto *offsets = cu_seqlens.ptr<int>();
    if (offsets[0] != 0 || offsets[batch] != ids.rows) throw std::invalid_argument("invalid recurrent sequence boundaries");
    int time = 0;
    for (int row = 0; row < batch; ++row) {
        if (offsets[row] > offsets[row + 1]) throw std::invalid_argument("unordered recurrent sequence boundaries");
        time = std::max(time, offsets[row + 1] - offsets[row]);
        lengths.ptr<int>()[size_t(row) * lengths.ld()] = 0;
    }
    auto state = Tensor::zeros({batch, encoder_.hidden()});
    auto workspace = Tensor::empty({batch, 6 * encoder_.hidden()});
    auto selected = Tensor::empty({batch, 1}, Type::I32);
    auto active = Tensor::empty({batch, 1}, Type::I32);
    auto embedded = Tensor::empty({batch, std::max(encoder_embedding_.cols, decoder_embedding_.cols)});
    auto encode_input = embedded.view().columns(0, encoder_embedding_.cols);
    auto decode_input = embedded.view().columns(0, decoder_embedding_.cols);
    for (int step = 0; step <= time; ++step) {
        for (int row = 0; row < batch; ++row) {
            const int count = offsets[row + 1] - offsets[row];
            selected.data_ptr<int>()[row] = step < count ? ids.ptr<int>()[offsets[row] + step] : encoder_end_;
            active.data_ptr<int>()[row] = step <= count;
        }
        ops::gather_rows(encoder_embedding_, selected.view(), encode_input);
        encoder_.forward(encode_input, state.view(), workspace.view(), active.view());
    }
    std::fill_n(selected.data_ptr<int>(), batch, decoder_start_);
    std::fill_n(active.data_ptr<int>(), batch, 1);
    auto output = Tensor::zeros({batch, max_steps_}, Type::I32);
    for (int step = 0, pending = batch; step < max_steps_ && pending; ++step) {
        ops::gather_rows(decoder_embedding_, selected.view(), decode_input);
        decoder_.forward(decode_input, state.view(), workspace.view(), active.view());
        ops::project_argmax(state.view(), output_, output_bias_, selected.view());
        for (int row = 0; row < batch; ++row) if (active.data_ptr<int>()[row]) {
            const int token = selected.data_ptr<int>()[row];
            if (token == decoder_end_) { active.data_ptr<int>()[row] = 0; --pending; }
            else {
                auto &count = lengths.ptr<int>()[size_t(row) * lengths.ld()];
                output.data_ptr<int>()[size_t(row) * max_steps_ + count++] = token;
            }
        }
    }
    return output;
}
} // namespace tts
