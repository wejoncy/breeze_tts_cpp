#pragma once
#include "core/tensor.h"
#include "core/allocator.h"

namespace tts {
class Buffer {
    void *data_ = nullptr;
    size_t size_ = 0;
    BufferAllocator *allocator_ = nullptr;
    int device_ = -1;

  public:
    Buffer() = default;
    Buffer(size_t n, BufferAllocator *allocator)
        : size_(n), allocator_(allocator), device_(allocator ? allocator->device_ordinal() : -1) {
        if (n) {
            if (!allocator_)
                throw std::invalid_argument("Buffer requires an explicit allocator");
            data_ = allocator_->allocate(n);
        }
    }
    ~Buffer() {
        if (data_)
            allocator_->deallocate(data_);
    }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    Buffer(Buffer &&o) noexcept
        : data_(o.data_), size_(o.size_), allocator_(o.allocator_), device_(o.device_) {
        o.data_ = nullptr;
        o.size_ = 0;
    }
    Buffer &operator=(Buffer &&o) noexcept {
        if (this != &o) {
            if (data_)
                allocator_->deallocate(data_);
            data_ = o.data_;
            size_ = o.size_;
            allocator_ = o.allocator_;
            device_ = o.device_;
            o.data_ = nullptr;
            o.size_ = 0;
        }
        return *this;
    }
    void *data() const {
        return data_;
    }
    size_t size() const {
        return size_;
    }
    int device() const { return device_; }
    TensorView view(int rows, int cols, Type type = Type::BF16, size_t offset = 0,
                    int stride = 0) const {
        if (offset > size_ || offset % bytes(type))
            throw std::runtime_error("device buffer bounds");
        auto v = TensorView::metadata(rows, cols, type, stride, device_);
        if (data_) v.data = static_cast<char *>(data_) + offset;
        if (v.span_bytes() > size_ - offset)
            throw std::runtime_error("device buffer view exceeds storage");
        return v;
    }
    template <typename T> T *ptr() const {
        return static_cast<T *>(data_);
    }
};
} // namespace tts
