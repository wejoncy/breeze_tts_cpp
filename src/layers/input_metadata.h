#pragma once
#include "core/forward_context.h"
#include "core/sequence_builder.h"

namespace tts {
inline TensorView upload_metadata(ForwardContext &context, const void *source,
                                   TensorShape shape, Type type) {
    auto result = context.tensor(shape, type);
    if (context.mode == ForwardContext::Execute && result.size())
        result.copy_({const_cast<void *>(source), shape.rows, shape.cols, type}, true);
    return result;
}
inline SequenceLayout upload_sequence(ForwardContext &context, const SequenceBuilder &builder) {
    const auto source = builder.view();
    if (builder.uniform()) return SequenceLayout::uniform(source.sequences, source.max_length);
    std::vector<int> values(size_t(source.rows) * 2 + size_t(source.sequences) * 2 + 1);
    auto *destination = values.data();
    for (auto span : {std::pair{source.row_batch, size_t(source.rows)},
                      std::pair{source.row_position, size_t(source.rows)},
                      std::pair{source.cu_seqlens, size_t(source.sequences) + 1},
                      std::pair{source.lengths, size_t(source.sequences)}}) {
        std::copy_n(span.first, span.second, destination);
        destination += span.second;
    }
    const auto metadata = upload_metadata(context, values.data(), {int(values.size()), 1}, Type::I32);
    const auto *pointer = metadata.ptr<int>();
    return {pointer, pointer + source.rows, pointer + size_t(source.rows) * 2,
        pointer + size_t(source.rows) * 2 + source.sequences + 1,
        source.rows, source.sequences, source.max_length};
}
} // namespace tts
