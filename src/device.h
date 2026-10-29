#pragma once
#include "tensor.h"
#include "memory_stats.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace tts {
inline void check(cudaError_t s, const char *expr) {
    if (s != cudaSuccess)
        throw std::runtime_error(std::string(expr) + ": " + cudaGetErrorString(s));
}
#define CUDA(x) ::tts::check((x), #x)
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
    explicit DeviceContext(int ordinal) : device(ordinal) {
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
        cudaMemPoolDestroy(pool_);
        cudaStreamDestroy(stream);
        cudaSetDevice(previous);
    }
    DeviceContext(const DeviceContext &) = delete;
    DeviceContext &operator=(const DeviceContext &) = delete;
    void *allocate(size_t size) override {
        if (!size)
            return nullptr;
        DeviceScope scope(device);
        cudaStreamCaptureStatus status;
        CUDA(cudaStreamIsCapturing(stream, &status));
        if (status != cudaStreamCaptureStatusNone)
            throw std::logic_error("device allocation during graph capture");
        void *data = nullptr;
        auto error = cudaMallocFromPoolAsync(&data, size, pool_, stream);
        if (error != cudaSuccess) {
            cudaGetLastError();
            throw std::runtime_error("CUDA pool cannot allocate " + std::to_string(size) +
                                     " bytes: " + cudaGetErrorString(error));
        }
        return data;
    }
    void deallocate(void *data) noexcept override {
        if (!data)
            return;
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
// Request-local bump allocations borrow a stable block from the shared pool.
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
    cudaStream_t stream;
    ArenaLease(DeviceArena &a, cudaStream_t s) : arena(a), stream(s) {
        arena.reset();
    }
    ArenaLease(const ArenaLease &) = delete;
    ArenaLease &operator=(const ArenaLease &) = delete;
    ~ArenaLease() {
        cudaStreamSynchronize(stream);
        arena.reset();
    }
};
} // namespace tts
