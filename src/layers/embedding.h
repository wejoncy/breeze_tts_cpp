#pragma once
#include "core/forward_context.h"
#include "ops/basic.h"
#include "ops/audio.h"

namespace tts {
class Embedding {
    TensorView weight_, replacement_;
    float scale_;
    int replacement_id_;
  public:
    explicit Embedding(TensorView weight, float scale = 1, TensorView replacement = {},
                        int replacement_id = -1)
        : weight_(weight), replacement_(replacement), scale_(scale), replacement_id_(replacement_id) {}
    TensorView forward(TensorView input_ids, ForwardContext &context) const {
        const int rows = context.batch.rows();
        if (input_ids.type != Type::I32 || input_ids.numel() < size_t(rows))
            throw std::invalid_argument("embedding input shape/type");
        const int output_rows = context.mode == ForwardContext::Prepare ? 1 : context.max_rows;
        auto output = context.tensor({output_rows, weight_.cols}, weight_.type)
                          .slice(0, context.mode == ForwardContext::Prepare ? 1 : rows);
        if (context.mode == ForwardContext::Execute)
            ops::lookup(weight_, input_ids.ptr<int>(), output, scale_, replacement_, replacement_id_);
        return output;
    }
    TensorView forward_codebooks(TensorView code_rows, TensorView owner_rows, int codebooks,
                                 ForwardContext &context) const {
        auto output = context.tensor({context.max_rows, weight_.cols}, weight_.type)
                          .slice(0, context.batch.rows());
        if (context.mode == ForwardContext::Execute)
            audio_embedding(weight_, nullptr, output, context.batch.rows(), 1, codebooks, -1,
                              code_rows.ptr<const int *>(), owner_rows.ptr<int>());
        return output;
    }
    TensorView forward_codebooks(ArrayView<const TensorRegion> regions, TensorView destination,
                                 int codebooks, ForwardContext &context) const {
        if (context.mode != ForwardContext::Execute) return destination;
        for (size_t region_index = 0; region_index < regions.size; ++region_index) {
            const auto &region = regions[region_index];
            if (region.tensor.type != Type::I32 || region.tensor.cols != codebooks)
                throw std::invalid_argument("codebook embedding region shape/type mismatch");
            auto output = destination.slice(region.first_row, region.tensor.rows);
            audio_embedding(weight_, region.tensor.ptr<int>(), output, output.rows, 1, codebooks);
        }
        return destination;
    }
};
} // namespace tts
