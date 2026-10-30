#pragma once
#include "runtime/device.h"
#include "core/kv_cache_view.h"
#include <algorithm>
#include <climits>
#include <vector>

namespace tts {
// Engine-wide physical pages. Appended chunks never relocate live K/V storage.
// All users execute on memory.stream; returning pages needs no host synchronization.
class PagedKVCache {
    DeviceContext &memory_;
    int layers_, width_, max_lanes_, max_blocks_, context_, page_size_, committed_ = 0;
    Type type_;
    size_t layer_bytes_, page_bytes_;
    Buffer key_table_, value_table_;
    std::vector<Buffer> chunks_;
    std::vector<void *> keys_, values_;
    std::vector<int> free_;
    static int blocks(int layers, int width, int lanes, int context, int page) {
        if (layers <= 0 || width <= 0 || lanes <= 0 || context <= 0 ||
            (page != 64 && page != 128 && page != 256) ||
            size_t(lanes) * ((size_t(context) + page - 1) / page) > INT_MAX)
            throw std::invalid_argument("invalid KV pool dimensions");
        return int((size_t(context) + page - 1) / page);
    }

  public:
    PagedKVCache(DeviceContext &memory, int layers, int width, Type type, int lanes, int context,
                 int page_tokens = 128)
        : memory_(memory), layers_(layers), width_(width), max_lanes_(lanes),
          max_blocks_(blocks(layers, width, lanes, context, page_tokens)), context_(context),
          page_size_(page_tokens), type_(type),
          layer_bytes_(size_t(page_tokens) * width * bytes(type)),
          page_bytes_(size_t(layers) * layer_bytes_),
          key_table_(size_t(lanes) * max_blocks_ * sizeof(void *), &memory),
          value_table_(key_table_.size(), &memory) {
        keys_.reserve(size_t(lanes) * max_blocks_);
        values_.reserve(keys_.capacity());
        free_.reserve(keys_.capacity());
    }
    int capacity() const {
        return int(keys_.size());
    }
    int available() const {
        return int(free_.size());
    }
    int committed() const {
        return committed_;
    }
    size_t additional_bytes(const std::vector<int> &limits) const {
        return additional_bytes(required_pages(limits));
    }
    size_t additional_bytes(int pages) const {
        if (pages < 0 || pages > max_lanes_ * max_blocks_ - committed_)
            throw std::invalid_argument("KV commitments exceed pool limits");
        return size_t(std::max(0, committed_ + pages - capacity())) * page_bytes_ * 2;
    }
    int block_stride() const {
        return max_blocks_;
    }
    size_t storage_capacity_bytes() const {
        return size_t(capacity()) * page_bytes_ * 2;
    }
    size_t max_storage_bytes() const {
        return size_t(max_lanes_) * max_blocks_ * page_bytes_ * 2;
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
        if (n <= capacity())
            return;
        int old = capacity(), added = n - old;
        if (page_bytes_ > SIZE_MAX / 2 / size_t(added))
            throw std::overflow_error("KV storage byte count");
        Buffer chunk(2 * page_bytes_ * added, &memory_);
        chunks_.push_back(std::move(chunk));
        auto *base = static_cast<char *>(chunks_.back().data());
        for (int i = 0; i < added; ++i) {
            keys_.push_back(base + size_t(i) * page_bytes_ * 2);
            values_.push_back(base + (size_t(i) * 2 + 1) * page_bytes_);
            free_.push_back(old + i);
        }
        CUDA(cudaMemcpyAsync(key_table_.ptr<void *>() + old, keys_.data() + old,
                             added * sizeof(void *), cudaMemcpyHostToDevice, memory_.stream));
        CUDA(cudaMemcpyAsync(value_table_.ptr<void *>() + old, values_.data() + old,
                             added * sizeof(void *), cudaMemcpyHostToDevice, memory_.stream));
        CUDA(cudaStreamSynchronize(memory_.stream));
    }
    KVCacheView view(const int *table = nullptr) const {
        CacheLayout layout{table, max_blocks_, page_size_};
        layout.key_pages = key_table_.ptr<void *>();
        layout.value_pages = value_table_.ptr<void *>();
        layout.page_layer_bytes = layer_bytes_;
        return {{nullptr, 0, width_, type_}, {nullptr, 0, width_, type_}, 0, layout};
    }
    // Request-wide credits cover any simultaneously active subset of its inputs.
    // Credits do not own an execution context or a private page pool.
    class Reservation {
        friend class Lease;
        PagedKVCache &pool_;
        int pages_, available_;

      public:
        Reservation(PagedKVCache &pool, int pages) : pool_(pool), pages_(pages), available_(pages) {
            pool.additional_bytes(pages);
            pool.reserve(pool.committed_ + pages);
            pool.committed_ += pages;
        }
        ~Reservation() {
            if (available_ != pages_)
                std::terminate();
            pool_.committed_ -= pages_;
        }
        Reservation(const Reservation &) = delete;
        void acquire(int pages) {
            if (pages > available_)
                throw std::logic_error("KV request reservation exhausted");
            available_ -= pages;
        }
        void release(int pages) noexcept {
            available_ += pages;
        }
        bool belongs_to(const PagedKVCache &pool) const {
            return &pool_ == &pool;
        }
    };
    class Lease {
        PagedKVCache &pool_;
        int *table_;
        std::vector<int> pages_, counts_, mapping_, limits_;
        std::unique_ptr<Reservation> owned_;
        Reservation *reservation_;
        int committed_ = 0;

      public:
        Lease(PagedKVCache &pool, const std::vector<int> &lengths, const std::vector<int> &limits,
              int *table, Reservation *reservation = nullptr)
            : pool_(pool), table_(table), counts_(lengths.size()),
              mapping_(lengths.size() * pool.max_blocks_, -1), limits_(limits),
              reservation_(reservation) {
            if (!table)
                throw std::invalid_argument("KV lease needs a request page table");
            pool.required_pages(lengths);
            if (lengths.size() != limits.size())
                throw std::invalid_argument("KV limit lane count changed");
            for (size_t i = 0; i < lengths.size(); ++i)
                if (lengths[i] > limits[i])
                    throw std::invalid_argument("KV length exceeds its admission limit");
            pages_.reserve(mapping_.size());
            int commitment = pool.required_pages(limits);
            if (!reservation_) {
                owned_ = std::make_unique<Reservation>(pool, commitment);
                reservation_ = owned_.get();
            }
            if (!reservation_->belongs_to(pool))
                throw std::invalid_argument("KV reservation belongs to another pool");
            reservation_->acquire(commitment);
            committed_ = commitment;
            try {
                ensure(lengths);
            } catch (...) {
                release();
                throw;
            }
        }
        ~Lease() {
            release();
        }
        Lease(const Lease &) = delete;
        KVCacheView view() const {
            return pool_.view(table_);
        }
        const std::vector<int> &pages() const {
            return pages_;
        }
        const std::vector<int> &block_table() const {
            return mapping_;
        }
        void ensure(const std::vector<int> &lengths) {
            pool_.required_pages(lengths);
            if (lengths.size() != counts_.size())
                throw std::invalid_argument("KV lane count changed");
            int extra = 0;
            for (size_t i = 0; i < lengths.size(); ++i) {
                if (lengths[i] > limits_[i])
                    throw std::invalid_argument("KV length exceeds its admission limit");
                extra += std::max(0, (lengths[i] + pool_.page_size_ - 1) / pool_.page_size_ -
                                         counts_[i]);
            }
            if (!extra)
                return;
            // Admission reserved every possible page; decode only assigns free page IDs.
            if (extra > pool_.available())
                throw std::logic_error("KV admission guarantee violated");
            for (size_t lane = 0; lane < lengths.size(); ++lane) {
                int needed = (lengths[lane] + pool_.page_size_ - 1) / pool_.page_size_;
                for (int block = counts_[lane]; block < needed; ++block) {
                    int page = pool_.free_.back();
                    pool_.free_.pop_back();
                    pages_.push_back(page);
                    mapping_[lane * pool_.max_blocks_ + block] = page;
                }
                counts_[lane] = std::max(counts_[lane], needed);
            }
            CUDA(cudaMemcpyAsync(table_, mapping_.data(), mapping_.size() * sizeof(int),
                                 cudaMemcpyHostToDevice, pool_.memory_.stream));
            CUDA(cudaStreamSynchronize(pool_.memory_.stream));
        }

      private:
        void release() noexcept {
            for (int page : pages_)
                pool_.free_.push_back(page);
            pages_.clear();
            reservation_->release(committed_);
            committed_ = 0;
        }
    };
};
} // namespace tts
