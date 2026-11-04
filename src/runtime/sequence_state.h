#pragma once
#include "runtime/device.h"
#include "core/sequence_state.h"

namespace tts {
// Request-persistent attention/convolution history. The model supplies sizes;
// storage and reset have no knowledge of its operators or checkpoint format.
struct SequenceState {
    Buffer keys, values, position;
    const size_t layer_bytes;
    std::vector<Buffer> history;
    SequenceState(const SequenceStateSpec &spec)
        : keys(spec.kv_bytes), values(spec.kv_bytes),
          position(sizeof(int)), layer_bytes(spec.layer_bytes) {
        history.reserve(spec.history_bytes.size());
        for (auto bytes : spec.history_bytes)
            history.emplace_back(bytes);
    }
    void reset(cudaStream_t stream) {
        CUDA(cudaMemsetAsync(position.data(), 0, position.size(), stream));
        for (auto &buffer : history)
            if (buffer.size())
                CUDA(cudaMemsetAsync(buffer.data(), 0, buffer.size(), stream));
    }
};
} // namespace tts
