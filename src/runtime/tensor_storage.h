#pragma once
#include "runtime/device.h"
namespace tts {
// Move-only owner; construction is explicit. Views borrow this owner's storage.
class Tensor {
    Buffer storage_;
    TensorShape shape_;
    Type type_ = Type::BF16;

  public:
    Tensor() = default;
    Tensor(TensorShape shape, Type type, BufferAllocator *arena)
        : storage_(checked_size(shape, type), arena), shape_(shape), type_(type) {}
    Tensor(const Tensor &) = delete;
    Tensor &operator=(const Tensor &) = delete;
    Tensor(Tensor &&) noexcept = default;
    Tensor &operator=(Tensor &&) noexcept = default;
    TensorView view() const {
        return storage_.view(shape_.rows, shape_.cols, type_);
    }

  private:
    static size_t checked_size(TensorShape s, Type t) {
        if (s.numel() > SIZE_MAX / bytes(t))
            throw std::overflow_error("tensor storage size");
        return s.numel() * bytes(t);
    }
};
} // namespace tts
