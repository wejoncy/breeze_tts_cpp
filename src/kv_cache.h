#pragma once
#include "device.h"
#include "kv_cache_view.h"
#include <algorithm>
#include <numeric>
#include <climits>
#include <vector>

namespace tts {
// Reusable physical pages. Current Engine admits one batched lease at a time.
// All layers of a page have the same owner. No contents need to survive idle growth.
class PagedKVCache {
    DeviceArena storage_, metadata_;
    Buffer keys_, values_, table_;
    int layers_, width_, max_lanes_, max_blocks_, context_, page_size_, pages_ = 0;
    Type type_;
    bool leased_ = false;
    std::vector<int> free_;
    static size_t table_bytes(int layers, int width, int lanes, int context, int page) {
        if (layers <= 0 || width <= 0 || lanes <= 0 || context <= 0 ||
            (page != 64 && page != 128 && page != 256))
            throw std::invalid_argument("invalid KV pool dimensions or page size");
        size_t blocks = (size_t(context) + page - 1) / page;
        // TensorView uses int row counts, including the flattened layer dimension.
        if (size_t(layers) * lanes > size_t(INT_MAX) / page / blocks)
            throw std::invalid_argument("KV pool exceeds tensor row index range");
        return size_t(lanes) * blocks * sizeof(int);
    }

  public:
    PagedKVCache(DeviceContext &memory, int layers, int width, Type type, int lanes,
                 int context, int page_tokens = 128)
        : storage_(memory),
          metadata_(memory, table_bytes(layers, width, lanes, context, page_tokens)),
          table_(table_bytes(layers, width, lanes, context, page_tokens), &metadata_),
          layers_(layers), width_(width), max_lanes_(lanes),
          max_blocks_((size_t(context) + page_tokens - 1) / page_tokens), context_(context),
          page_size_(page_tokens), type_(type) {}
    int capacity() const {
        return pages_;
    }
    size_t storage_capacity_bytes() const {
        return storage_.capacity();
    }
    int required_pages(const std::vector<int> &lengths) const {
        if (lengths.empty() || lengths.size() > size_t(max_lanes_))
            throw std::invalid_argument("invalid KV lane count");
        int n = 0;
        for (int length : lengths) {
            if (length <= 0 || length > context_)
                throw std::invalid_argument("invalid KV sequence capacity");
            n += (length + page_size_ - 1) / page_size_;
        }
        return n;
    }
    void reserve(int n) {
        if (n < 0 || n > max_lanes_ * max_blocks_)
            throw std::invalid_argument("KV page reservation exceeds pool limits");
        if (n <= pages_)
            return;
        if (leased_)
            throw std::logic_error("cannot grow leased KV cache");
        // Retain the high-water mark, but do not round by an arbitrary page group.
        size_t one = size_t(layers_) * n * page_size_ * width_ * bytes(type_);
        if (one > SIZE_MAX / 2 - 255)
            throw std::overflow_error("KV storage byte count");
        keys_ = Buffer();
        values_ = Buffer();
        pages_ = 0;
        free_.clear();
        storage_.reset();
        storage_.reserve(2 * BufferAllocator::aligned(one));
        keys_ = Buffer(one, &storage_);
        values_ = Buffer(one, &storage_);
        pages_ = n;
        free_.resize(n);
        std::iota(free_.rbegin(), free_.rend(), 0);
    }
    TensorView keys(int layer) const {
        if (layer < 0 || layer >= layers_)
            throw std::out_of_range("KV layer");
        return keys_.view(pages_ * page_size_, width_, type_,
                          size_t(layer) * pages_ * page_size_ * width_ * bytes(type_));
    }
    TensorView values(int layer) const {
        if (layer < 0 || layer >= layers_)
            throw std::out_of_range("KV layer");
        return values_.view(pages_ * page_size_, width_, type_,
                            size_t(layer) * pages_ * page_size_ * width_ * bytes(type_));
    }
    CacheLayout layout() const {
        return {table_.ptr<int>(), max_blocks_, page_size_, false};
    }
    KVCacheView view() const {
        return {keys_.view(layers_ * pages_ * page_size_, width_, type_),
                values_.view(layers_ * pages_ * page_size_, width_, type_), pages_ * page_size_,
                layout()};
    }
    class Lease {
        PagedKVCache &pool_;
        cudaStream_t stream_;
        std::vector<int> pages, mapping;

      public:
        Lease(PagedKVCache &pool, cudaStream_t stream, const std::vector<int> &lengths)
            : pool_(pool), stream_(stream) {
            int needed = pool.required_pages(lengths);
            if (pool.leased_ || lengths.size() > size_t(pool.max_lanes_) ||
                needed > int(pool.free_.size()))
                throw std::runtime_error("KV cache admission failed");
            pages.reserve(needed);
            mapping.assign(lengths.size() * pool.max_blocks_, -1);
            for (size_t lane = 0; lane < lengths.size(); ++lane)
                for (int j = 0; j < (lengths[lane] + pool.page_size_ - 1) / pool.page_size_; ++j) {
                    int page = pool.free_.back();
                    pool.free_.pop_back();
                    pages.push_back(page);
                    mapping[lane * pool.max_blocks_ + j] = page;
                }
            pool.leased_ = true;
            try {
                CUDA(cudaMemcpyAsync(pool.table_.data(), mapping.data(), mapping.size() * 4,
                                     cudaMemcpyHostToDevice, stream));
                CUDA(cudaStreamSynchronize(stream));
            } catch (...) {
                cudaStreamSynchronize(stream_);
                release();
                throw;
            }
        }
        ~Lease() {
            cudaStreamSynchronize(stream_);
            release();
        }
        Lease(const Lease &) = delete;

      private:
        void release() noexcept {
            for (int page : pages)
                pool_.free_.push_back(page);
            pages.clear();
            pool_.leased_ = false;
        }
    };
};
} // namespace tts
