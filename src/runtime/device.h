#pragma once
#include "core/tensor.h"
#include "tts/memory_stats.h"
#include "runtime/block_pool.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <memory>
#include <initializer_list>
#include <unordered_map>

namespace tts {
struct DeviceError : std::runtime_error {
    using std::runtime_error::runtime_error;
};
inline void check(cudaError_t s, const char *expr) {
    if (s != cudaSuccess)
        throw DeviceError(std::string(expr) + ": " + cudaGetErrorString(s));
}
#define CUDA(x) ::tts::check((x), #x)
struct DeviceOutOfMemory : DeviceError {
    using DeviceError::DeviceError;
};
// Storage sizing and allocation use the same constructors and alignment rules.
struct BufferAllocator {
    virtual ~BufferAllocator() = default;
    virtual void *allocate(size_t size) = 0;
    virtual void deallocate(void *) noexcept = 0;
    static size_t aligned(size_t size) {
        if (size > SIZE_MAX - 255)
            throw std::overflow_error("arena allocation size");
        return (size + 255) & ~size_t(255);
    }
};
struct WorkspacePlan : BufferAllocator {
    void deallocate(void *) noexcept override {}
    size_t size = 0;
    void *allocate(size_t n) override {
        n = aligned(n);
        if (n > SIZE_MAX - size)
            throw std::overflow_error("workspace size");
        size += n;
        return nullptr; // metadata-only buffers; never submitted to a backend
    }
};
// One execution stream and its pool. Model, KV, workspaces and tuning buffers
// share this resource; allocator references never own or extend its lifetime.
class DeviceContext final : public BufferAllocator {
    cudaMemPool_t pool_ = nullptr;
    void *runtime_base_ = nullptr;
    size_t runtime_bytes_ = 0;
    std::unique_ptr<BlockPool> runtime_;
    size_t memory_limit_ = 0, live_bytes_ = 0;
    std::unordered_map<void *, size_t> allocations_;
    class DeviceScope {
        int previous_ = -1;

      public:
        explicit DeviceScope(int device) {
            CUDA(cudaGetDevice(&previous_));
            if (previous_ != device)
                CUDA(cudaSetDevice(device));
        }
        ~DeviceScope() {
            if (previous_ >= 0)
                cudaSetDevice(previous_);
        }
    };

  public:
    const int device;
    cudaStream_t stream = nullptr;
    explicit DeviceContext(int ordinal, size_t memory_limit = 0)
        : memory_limit_(memory_limit), device(ordinal) {
        DeviceScope scope(device);
        int supported = 0;
        CUDA(cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported, device));
        if (!supported)
            throw std::runtime_error("CUDA device does not support memory pools");
        CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        cudaMemPoolProps props{};
        props.allocType = cudaMemAllocationTypePinned;
        props.location.type = cudaMemLocationTypeDevice;
        props.location.id = device;
        try {
            CUDA(cudaMemPoolCreate(&pool_, &props));
            // Live arena reservations remain reusable. Freed load/tuning scratch
            // may return to the device at synchronization instead of pinning VRAM.
            uint64_t retain = 0;
            CUDA(cudaMemPoolSetAttribute(pool_, cudaMemPoolAttrReleaseThreshold, &retain));
        } catch (...) {
            if (pool_)
                cudaMemPoolDestroy(pool_);
            cudaStreamDestroy(stream);
            throw;
        }
    }
    ~DeviceContext() {
        int previous = device;
        cudaGetDevice(&previous);
        cudaSetDevice(device);
        cudaStreamSynchronize(stream);
        if (runtime_base_) {
            cudaFreeAsync(runtime_base_, stream);
            cudaStreamSynchronize(stream);
        }
        cudaMemPoolDestroy(pool_);
        cudaStreamDestroy(stream);
        cudaSetDevice(previous);
    }
    DeviceContext(const DeviceContext &) = delete;
    DeviceContext &operator=(const DeviceContext &) = delete;
    // Called once after loading, planning and capture. Every subsequent Buffer
    // allocation borrows fixed backing, including eager activations and KV chunks.
    void reserve_runtime(size_t size) {
        if (runtime_ || !size)
            throw std::logic_error("invalid runtime reservation");
        size = aligned(size);
        auto blocks = std::make_unique<BlockPool>(size);
        void *base = allocate(size);
        sync();
        runtime_bytes_ = size;
        runtime_base_ = base;
        runtime_ = std::move(blocks);
    }
    void *allocate(size_t size) override {
        if (!size)
            return nullptr;
        DeviceScope scope(device);
        cudaStreamCaptureStatus status;
        CUDA(cudaStreamIsCapturing(stream, &status));
        if (status != cudaStreamCaptureStatusNone)
            throw std::logic_error("device allocation during graph capture");
        if (runtime_) {
            try {
                return static_cast<char *>(runtime_base_) + runtime_->allocate(aligned(size));
            } catch (const std::bad_alloc &) {
                throw DeviceOutOfMemory("initialized runtime pool exhausted: needs " +
                                        std::to_string(size) + " bytes, largest free region " +
                                        std::to_string(runtime_->largest()) + " bytes");
            }
        }
        if (memory_limit_ && size > memory_limit_ - live_bytes_)
            throw DeviceOutOfMemory("engine memory budget exceeded: requested " +
                                    std::to_string(size) + " bytes, remaining " +
                                    std::to_string(memory_limit_ - live_bytes_) + " bytes");
        void *data = nullptr;
        auto error = cudaMallocFromPoolAsync(&data, size, pool_, stream);
        if (error != cudaSuccess) {
            cudaGetLastError();
            std::string message = "CUDA pool cannot allocate " + std::to_string(size) +
                                  " bytes: " + cudaGetErrorString(error);
            if (error == cudaErrorMemoryAllocation)
                throw DeviceOutOfMemory(message);
            throw DeviceError(message);
        }
        try {
            allocations_.emplace(data, size);
        } catch (...) {
            cudaFreeAsync(data, stream);
            throw;
        }
        live_bytes_ += size;
        return data;
    }
    void deallocate(void *data) noexcept override {
        if (!data)
            return;
        auto offset = uintptr_t(data) - uintptr_t(runtime_base_);
        if (runtime_ && offset < runtime_bytes_) {
            runtime_->deallocate(size_t(offset));
            return;
        }
        auto allocation = allocations_.find(data);
        if (allocation == allocations_.end())
            std::terminate();
        live_bytes_ -= allocation->second;
        allocations_.erase(allocation);
        int previous = device;
        cudaGetDevice(&previous);
        cudaSetDevice(device);
        cudaFreeAsync(data, stream);
        cudaSetDevice(previous);
    }
    void sync() const {
        DeviceScope scope(device);
        CUDA(cudaStreamSynchronize(stream));
    }
    void trim() {
        sync();
        DeviceScope scope(device);
        CUDA(cudaMemPoolTrimTo(pool_, 0));
    }
    MemoryStats stats() const {
        DeviceScope scope(device);
        MemoryStats s{};
        CUDA(cudaMemPoolGetAttribute(pool_, cudaMemPoolAttrUsedMemCurrent, &s.used_bytes));
        CUDA(cudaMemPoolGetAttribute(pool_, cudaMemPoolAttrReservedMemCurrent, &s.reserved_bytes));
        CUDA(cudaMemPoolGetAttribute(pool_, cudaMemPoolAttrUsedMemHigh, &s.peak_used_bytes));
        CUDA(
            cudaMemPoolGetAttribute(pool_, cudaMemPoolAttrReservedMemHigh, &s.peak_reserved_bytes));
        return s;
    }
    size_t available_bytes() const {
        if (runtime_)
            return runtime_->available();
        DeviceScope scope(device);
        size_t free, total;
        CUDA(cudaMemGetInfo(&free, &total));
        const auto pool = stats();
        const size_t available = free + size_t(pool.reserved_bytes - pool.used_bytes);
        return memory_limit_ ? std::min(available, memory_limit_ - live_bytes_) : available;
    }
    size_t largest_available_bytes() const {
        return runtime_ ? runtime_->largest() : available_bytes();
    }
    bool can_allocate(std::initializer_list<size_t> sizes) const {
        if (!runtime_)
            throw std::logic_error("runtime pool has not been initialized");
        auto planned = *runtime_;
        try {
            for (auto size : sizes)
                if (size)
                    planned.allocate(aligned(size));
            return true;
        } catch (const std::bad_alloc &) {
            return false;
        }
    }
};
class Buffer {
    void *data_ = nullptr;
    size_t size_ = 0;
    BufferAllocator *allocator_ = nullptr;

  public:
    Buffer() = default;
    Buffer(size_t n, BufferAllocator *allocator) : size_(n), allocator_(allocator) {
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
    Buffer(Buffer &&o) noexcept : data_(o.data_), size_(o.size_), allocator_(o.allocator_) {
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
    TensorView view(int rows, int cols, Type type = Type::BF16, size_t offset = 0,
                    int stride = 0) const {
        if (offset > size_ || offset % bytes(type))
            throw std::runtime_error("device buffer bounds");
        TensorView v{data_ ? static_cast<char *>(data_) + offset : nullptr, rows, cols, type,
                     stride};
        if (v.span_bytes() > size_ - offset)
            throw std::runtime_error("device buffer view exceeds storage");
        return v;
    }
    template <typename T> T *ptr() const {
        return static_cast<T *>(data_);
    }
};
// Bump allocations borrow a stable block from the shared pool.
// All graph invalidation happens in ExecutionResources before this block grows.
class DeviceArena : public BufferAllocator {
    DeviceContext &memory_;
    Buffer storage_;
    size_t used_ = 0;

  public:
    explicit DeviceArena(DeviceContext &memory, size_t capacity = 0) : memory_(memory) {
        reserve(capacity);
    }
    DeviceArena(const DeviceArena &) = delete;
    void reserve(size_t n) {
        if (n <= capacity())
            return;
        if (used_)
            throw std::logic_error("cannot grow an active device arena");
        memory_.sync();
        storage_ = Buffer();
        storage_ = Buffer(aligned(n), &memory_);
        memory_.sync();
    }
    void *allocate(size_t size) override {
        size_t n = aligned(size);
        if (n > capacity() - used_)
            throw std::runtime_error("reserved device workspace exhausted");
        auto *base = static_cast<char *>(storage_.data());
        void *p = base ? base + used_ : nullptr;
        used_ += n;
        return p;
    }
    void deallocate(void *) noexcept override {}
    void reset() {
        used_ = 0;
    }
    size_t capacity() const {
        return storage_.size();
    }
};
struct ArenaLease {
    DeviceArena &arena;
    explicit ArenaLease(DeviceArena &a) : arena(a) {
        arena.reset();
    }
    ArenaLease(const ArenaLease &) = delete;
    ArenaLease &operator=(const ArenaLease &) = delete;
    // All uses are ordered on the arena's device stream. Storage growth/destruction
    // owns the synchronization; rewinding the CPU cursor does not free device bytes.
    ~ArenaLease() {
        arena.reset();
    }
};
} // namespace tts
