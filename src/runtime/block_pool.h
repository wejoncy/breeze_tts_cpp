#pragma once
#include <algorithm>
#include <cstddef>
#include <exception>
#include <new>
#include <stdexcept>
#include <vector>

namespace tts {
// Single-owner offset allocator. The owner orders all GPU uses on one stream.
// Returning and coalescing a region never allocates metadata or device storage.
class BlockPool {
    struct Block {
        size_t offset, size;
        bool free;
    };
    std::vector<Block> blocks_;
    size_t available_ = 0;

  public:
    explicit BlockPool(size_t size) : blocks_{{0, size, true}}, available_(size) {}
    size_t available() const {
        return available_;
    }
    size_t largest() const {
        size_t size = 0;
        for (const auto &block : blocks_)
            if (block.free)
                size = std::max(size, block.size);
        return size;
    }
    size_t allocate(size_t size) {
        if (!size)
            throw std::invalid_argument("zero-sized pool region");
        for (size_t i = 0; i < blocks_.size(); ++i) {
            const auto block = blocks_[i];
            if (!block.free || size > block.size)
                continue;
            if (size < block.size)
                blocks_.insert(blocks_.begin() + i + 1,
                               {block.offset + size, block.size - size, true});
            blocks_[i] = {block.offset, size, false};
            available_ -= size;
            return block.offset;
        }
        throw std::bad_alloc();
    }
    void deallocate(size_t offset) noexcept {
        for (size_t i = 0; i < blocks_.size(); ++i) {
            auto &block = blocks_[i];
            if (block.offset != offset || block.free)
                continue;
            available_ += block.size;
            block.free = true;
            if (i + 1 < blocks_.size() && blocks_[i + 1].free) {
                block.size += blocks_[i + 1].size;
                blocks_.erase(blocks_.begin() + i + 1);
            }
            if (i && blocks_[i - 1].free) {
                blocks_[i - 1].size += blocks_[i].size;
                blocks_.erase(blocks_.begin() + i);
            }
            return;
        }
        std::terminate(); // Internal ownership violation, never a request error.
    }
};
} // namespace tts
