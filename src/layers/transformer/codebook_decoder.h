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
        const int64_t layer_rows = int64_t(context.max_rows) * codebooks_;
        if (decoder_.config.layers < 1 || layer_rows < 1 || layer_rows > INT32_MAX / decoder_.config.layers ||
            context.max_rows > INT32_MAX / 2)
            throw std::invalid_argument("residual decoder cache exceeds tensor dimensions");
        KVCacheView cache;
        cache.rows_per_layer = int(layer_rows);
        cache.keys = context.tensor({int(layer_rows) * decoder_.config.layers,
            decoder_.config.kvheads * decoder_.config.dim}, decoder_.config.type);
        cache.values = context.tensor(cache.keys, decoder_.config.type);
        auto hidden_states = context.tensor({context.max_rows * 2, hidden_size}, decoder_.config.type);
        auto logits = context.tensor({context.max_rows, vocabulary_}, Type::F32).slice(0, lanes);
        auto first_states = hidden_states.slice(0, lanes * 2).every_rows(0, 2, lanes);
        auto token_states = hidden_states.slice(0, lanes * 2).every_rows(1, 2, lanes);
        projection_.forward(backbone_states, first_states, context);
        if (context.mode == ForwardContext::Execute)
            audio_embedding( embeddings_, nullptr, token_states, lanes, 1,
                             codebooks_, 0, code_rows.ptr<const int *>(), owner_rows.ptr<int>());
        auto decoder_context = context;
        decoder_context.max_batch = context.max_rows;
        decoder_context.max_time = codebooks_;
        decoder_context.max_rows = context.max_rows * 2;
        decoder_context.batch = ForwardBatch(lanes, 2);
        decoder_context.batch.fixed_position = 0;
        decoder_context.batch.cache = cache;
        decoder_.forward(hidden_states.slice(0, lanes * 2), decoder_context);
        for (int codebook_index = 1; codebook_index < codebooks_; ++codebook_index) {
            if (codebook_index > 1 && context.mode == ForwardContext::Execute) {
                token_states = hidden_states.slice(0, lanes);
                audio_embedding( embeddings_, nullptr, token_states, lanes, 1,
                                 codebooks_, codebook_index - 1, code_rows.ptr<const int *>(),
                                 owner_rows.ptr<int>());
                decoder_context.batch.sequence = SequenceLayout::uniform(lanes, 1);
                decoder_context.batch.fixed_position = codebook_index;
                decoder_.forward(token_states, decoder_context);
            }
            heads_[codebook_index - 1].forward(token_states, logits, context);
            sampler_.forward(logits, sampling_rows, done, context, codebook_index);
        }
        return done;
    }
};
} // namespace tts
