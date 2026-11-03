#pragma once
#include "runtime/device.h"

namespace tts {
struct SequenceStateSpec {
    size_t kv_bytes = 0;
    std::vector<size_t> history_bytes;
};
// Request-persistent attention/convolution history. The model supplies sizes;
// storage and reset have no knowledge of its operators or checkpoint format.
struct SequenceState {
    Buffer keys, values, position;
    std::vector<Buffer> history;
    SequenceState(const SequenceStateSpec &spec, BufferAllocator *storage)
        : keys(spec.kv_bytes, storage), values(spec.kv_bytes, storage),
          position(sizeof(int), storage) {
        history.reserve(spec.history_bytes.size());
        for (auto bytes : spec.history_bytes)
            history.emplace_back(bytes, storage);
    }
    void reset(cudaStream_t stream) {
        CUDA(cudaMemsetAsync(position.data(), 0, position.size(), stream));
        for (auto &buffer : history)
            if (buffer.size())
                CUDA(cudaMemsetAsync(buffer.data(), 0, buffer.size(), stream));
    }
};
} // namespace tts
