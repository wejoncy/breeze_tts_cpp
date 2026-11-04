#pragma once
#include "core/sequence_layout.h"
#include <cuda_runtime_api.h>

namespace tts {
struct SequenceAddress {
    int sequence, position, offset, extent, valid;
};
__device__ __forceinline__ SequenceAddress sequence_address(SequenceLayout p, int row, int time) {
    if (!p.cu_seqlens) {
        int b = row / time;
        return {b, row % time, b * time, time, time};
    }
    int b = p.row_batch ? p.row_batch[row] : 0;
    if (!p.row_batch) {
        int upper = p.sequences;
        while (b + 1 < upper) {
            int middle = b + (upper - b) / 2;
            if (p.cu_seqlens[middle] <= row) b = middle;
            else upper = middle;
        }
    }
    int first = p.cu_seqlens[b], extent = p.cu_seqlens[b + 1] - first;
    return {b, p.row_position ? p.row_position[row] : row - first, first, extent,
            p.lengths ? p.lengths[b] : extent};
}
} // namespace tts
