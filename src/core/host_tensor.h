#pragma once
#include "core/tensor_storage.h"
#include <cstring>

namespace tts {
class HostTensor {
    Tensor tensor_;

  public:
    HostTensor(TensorShape shape, Type type)
        : tensor_(shape, type) {
        if (tensor_.numel()) std::memset(tensor_.data_ptr(), 0, tensor_.view().size());
    }
    explicit HostTensor(TensorView source) : tensor_(source) {}
    TensorView view() const { return tensor_.view(); }
};
} // namespace tts
