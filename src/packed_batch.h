#pragma once
namespace tts {
// Borrowed device addressing metadata for variable-length sequences.
struct PackedBatch {
    const int *row_batch, *row_position, *offsets, *lengths;
    int rows;
};
} // namespace tts
