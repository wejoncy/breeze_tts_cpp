#pragma once
#include "layers/transformer/decoder.h"
#include "layers/linear.h"
#include "layers/sampler.h"
#include "ops/audio.h"

namespace tts {
// Residual codebooks use one temporary decoder cache for the whole head cascade.
class ResidualCodebooks {
    const Transformer &decoder_;
    TensorView embeddings_;
    Linear projection_;
    std::vector<Linear> heads_;
    Sampler sampler_;
    int codebooks_, vocabulary_;
  public:
    ResidualCodebooks(const Transformer &decoder, TensorView embeddings, TensorView projection,
                      const std::vector<TensorView> &heads, SamplingLayout sampling)
        : decoder_(decoder), embeddings_(embeddings), projection_(projection), sampler_(sampling),
          codebooks_(sampling.codebooks), vocabulary_(heads.empty() ? 0 : heads.front().rows) {
        if (heads.size() + 1 != size_t(codebooks_) || heads.empty() ||
            vocabulary_ < sampling.entries ||
            embeddings.cols != decoder.config.hidden ||
            projection_.output_channels() != decoder.config.hidden)
            throw std::invalid_argument("residual codebook parameter shapes");
        for (auto weight : heads) {
            heads_.emplace_back(weight);
            if (weight.cols != decoder.config.hidden || weight.rows != vocabulary_)
                throw std::invalid_argument("residual output head shape");
        }
    }
    TensorView forward(TensorView backbone_states, TensorView code_rows, TensorView owner_rows,
                        TensorView sampling_rows, TensorView done, ForwardContext &context) const {
        const int lanes = backbone_states.rows, hidden_size = decoder_.config.hidden;
        TransformerState workspace(decoder_.config, context.max_rows, codebooks_,
                                    context.max_rows * 2, &context.storage);
        auto hidden_states = context.tensor({context.max_rows * 2, hidden_size}, decoder_.config.type);
        auto logits = context.tensor({context.max_rows, vocabulary_}, decoder_.config.type).slice(0, lanes);
        auto first_states = hidden_states.slice(0, lanes * 2).every_rows(0, 2, lanes);
        auto token_states = hidden_states.slice(0, lanes * 2).every_rows(1, 2, lanes);
        if (context.mode == ForwardContext::Prepare && backbone_states.data) {
            decoder_.prepare(workspace, context.max_rows * 2);
            projection_.prepare_range(backbone_states.slice(0, 1), first_states.slice(0, 1),
                                       context.max_rows);
            for (const auto &head : heads_)
                head.prepare_range(token_states.slice(0, 1), logits.slice(0, 1), context.max_rows);
        } else if (context.mode == ForwardContext::Execute) {

            projection_.forward(backbone_states, first_states);
            audio_embedding( embeddings_, nullptr, token_states, lanes, 1,
                             codebooks_, 0, code_rows.ptr<const int *>(), owner_rows.ptr<int>());
            ForwardBatch decoder_batch(lanes, 2);
            decoder_batch.fixed_position = 0;
            decoder_.forward(workspace, hidden_states.slice(0, lanes * 2), decoder_batch);
            for (int codebook_index = 1; codebook_index < codebooks_; ++codebook_index) {
                if (codebook_index > 1) {
                    token_states = hidden_states.slice(0, lanes);
                    audio_embedding( embeddings_, nullptr, token_states, lanes, 1,
                                     codebooks_, codebook_index - 1, code_rows.ptr<const int *>(),
                                     owner_rows.ptr<int>());
                    decoder_batch.sequence = SequenceLayout::uniform(lanes, 1);
                    decoder_batch.fixed_position = codebook_index;
                    decoder_.forward(workspace, token_states, decoder_batch);
                }
                heads_[codebook_index - 1].forward(token_states, logits);
                sampler_.forward(logits, sampling_rows, done, context, codebook_index);
            }
        }
        return done;
    }
};
} // namespace tts
