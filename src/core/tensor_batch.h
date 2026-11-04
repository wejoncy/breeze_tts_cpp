#pragma once
#include "core/tensor.h"
#include <vector>

namespace tts {
struct TensorChunk {
    TensorView values;
    std::vector<int> rows;
    int time;
    int first_frame = 0, frame_count = 0;
};
struct TensorBatch {
    std::vector<TensorChunk> chunks;
    int rows = 0;
};
} // namespace tts
