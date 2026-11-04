#pragma once
#include "core/tensor.h"
#include <cstring>
#include <vector>

namespace tts {
class HostTensor {
    TensorShape shape_;
    Type type_;
    std::vector<uint64_t> storage_;

  public:
    HostTensor(TensorShape shape, Type type)
        : shape_(shape), type_(type), storage_((shape.numel() * bytes(type) + 7) / 8) {}
    explicit HostTensor(TensorView source) : HostTensor({source.rows, source.cols}, source.type) {
        if (source.device >= 0 || !source.dense() || (source.size() && !source.data))
            throw std::invalid_argument("host tensor requires dense CPU data");
        if (source.size()) std::memcpy(storage_.data(), source.data, source.size());
    }
    TensorView view() const {
        return TensorView(const_cast<uint64_t *>(storage_.data()), shape_.rows, shape_.cols, type_);
    }
};
} // namespace tts
