#pragma once
#include "core/alignment.h"
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <exception>
#include <algorithm>
#include <memory>

namespace tts {
struct BufferAllocator {
    virtual ~BufferAllocator() = default;
    virtual void *allocate(size_t bytes) = 0;
    virtual void deallocate(void *) noexcept = 0;
    virtual int device_ordinal() const { return -1; }
    virtual std::shared_ptr<BufferAllocator> lifetime() { return {}; }
};
// Forward tensors borrow storage until the enclosing forward scope ends.
struct ScratchAllocator : BufferAllocator {
    void deallocate(void *) noexcept final {}
    virtual size_t position() const noexcept = 0;
    virtual void rewind(size_t position) noexcept = 0;
};
struct WorkspacePlan final : ScratchAllocator {
  private:
    size_t cursor_ = 0;

  public:
    size_t size = 0;
    void *allocate(size_t bytes) override {
        bytes = aligned_storage_bytes(bytes);
        if (bytes > SIZE_MAX - cursor_) throw std::overflow_error("workspace size");
        cursor_ += bytes;
        size = std::max(size, cursor_);
        return nullptr;
    }
    size_t position() const noexcept override { return cursor_; }
    void rewind(size_t position) noexcept override {
        if (position > cursor_) std::terminate();
        cursor_ = position;
    }
};
} // namespace tts
