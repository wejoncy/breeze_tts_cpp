#pragma once
#include "core/tensor.h"
#include "core/allocator.h"
#include "core/buffer.h"
#include "tts/memory_stats.h"
#include "runtime/block_pool.h"
#include "runtime/error.h"
#include "ops/cuda/cuda_check.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <memory>
#include <initializer_list>
#include <unordered_map>
#include <cstring>

namespace tts {
// One execution stream and its pool. Model, KV, workspaces and tuning buffers
// share this resource. Owning tensors retain it until their storage is released.
class DeviceContext final : public BufferAllocator, public std::enable_shared_from_this<DeviceContext> {
    class HostPool final : public BufferAllocator {
        DeviceContext &context_;
        void *base_ = nullptr;
        size_t bytes_ = 0;
        std::unique_ptr<BlockPool> blocks_;
      public:
        explicit HostPool(DeviceContext &context) : context_(context) {}
        void reserve(size_t bytes) {
            auto blocks = std::make_unique<BlockPool>(bytes);
            CUDA(cudaHostAlloc(&base_, bytes, cudaHostAllocDefault));
            bytes_ = bytes;
            blocks_ = std::move(blocks);
        }
        void release() { if (base_) cudaFreeHost(base_); }
        bool ready() const { return bool(blocks_); }
        bool owns(const void *data) const {
            return base_ && uintptr_t(data) - uintptr_t(base_) < bytes_;
        }
        std::shared_ptr<BufferAllocator> lifetime() override { return context_.lifetime(); }
        void *allocate(size_t bytes) override {
            if (!blocks_) throw std::logic_error("pinned tensor pool is not initialized");
            return static_cast<char *>(base_) + blocks_->allocate(Tensor::aligned(bytes));
        }
        void deallocate(void *data) noexcept override {
            // A CPU allocation cannot be reused until its DMA has finished.
            try { context_.sync(); } catch (...) {}
            blocks_->deallocate(uintptr_t(data) - uintptr_t(base_));
        }
    } host_{*this};
    cudaMemPool_t pool_ = nullptr;
    void *runtime_base_ = nullptr;
    size_t runtime_bytes_ = 0;
    std::unique_ptr<BlockPool> runtime_;
    size_t memory_limit_ = 0, live_bytes_ = 0;
    std::unordered_map<void *, size_t> allocations_;
    void *transfer_base_ = nullptr;
    size_t transfer_bytes_ = 0;
    mutable size_t transfer_used_ = 0;
    void *transfer(size_t bytes) const {
        size_t n = Tensor::aligned(bytes);
        if (n > transfer_bytes_)
            throw std::logic_error("initialized host transfer arena exhausted");
        // Usually reset at the runner's existing completion checkpoint. A burst
        // of rejected/cancelled preparations can instead fill the fixed arena;
        // finish its transfers before reusing backing, without growing resources.
        if (n > transfer_bytes_ - transfer_used_) sync();
        auto *data = static_cast<char *>(transfer_base_) + transfer_used_;
        transfer_used_ += n;
        return data;
    }
    class DeviceScope {
        int previous_ = -1;

      public:
        explicit DeviceScope(int device) {
            CUDA(cudaGetDevice(&previous_));
            if (previous_ != device)
                CUDA(cudaSetDevice(device));
            else
                previous_ = -1;
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
        host_.release();
        if (transfer_base_)
            cudaFreeHost(transfer_base_);
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
    int device_ordinal() const override { return device; }
    std::shared_ptr<BufferAllocator> lifetime() override { return weak_from_this().lock(); }
    BufferAllocator &host_allocator() { return host_; }
    bool has_host_tensor_backing() const { return host_.ready(); }
    // Init-only pinned backing. Uploads retain a snapshot until stream completion;
    // neither temporary CPU metadata nor request cancellation can invalidate it.
    void reserve_transfers(size_t bytes) {
        if (!bytes || transfer_base_ || runtime_)
            throw std::logic_error("invalid host transfer reservation");
        DeviceScope scope(device);
        CUDA(cudaHostAlloc(&transfer_base_, Tensor::aligned(bytes), cudaHostAllocDefault));
        transfer_bytes_ = Tensor::aligned(bytes);
        host_.reserve(transfer_bytes_);
    }
    bool has_transfer_backing() const { return transfer_base_ != nullptr; }
    void copy(TensorView source, TensorView destination, bool non_blocking) {
        if (source.type != destination.type ||
            source.rows != destination.rows || source.cols != destination.cols ||
            (source.device >= 0 && source.device != device) ||
            (destination.device >= 0 && destination.device != device))
            throw std::invalid_argument("tensor transfer requires matching storage on the initialized device");
        if (!source.size()) return;
        if (!source.data || !destination.data) throw std::invalid_argument("tensor transfer has no storage");
        if (non_blocking && destination.device < 0 && !host_.owns(destination.data))
            throw std::invalid_argument("non-blocking CPU copy requires pinned storage; use to(cpu, true)");
        DeviceScope scope(device);
        const void *input = source.data;
        size_t pitch = size_t(source.ld()) * bytes(source.type);
        if (source.device < 0 && destination.device >= 0 && !host_.owns(source.data)) {
            if (non_blocking && !transfer_base_)
                throw std::logic_error("non-blocking transfer requires initialized host backing");
            if (transfer_base_) {
                auto *snapshot = static_cast<char *>(transfer(source.size()));
                for (int row = 0; row < source.rows; ++row)
                    std::memcpy(snapshot + size_t(row) * source.cols * bytes(source.type),
                        static_cast<const char *>(source.data) + size_t(row) * pitch,
                        size_t(source.cols) * bytes(source.type));
                input = snapshot;
                pitch = size_t(source.cols) * bytes(source.type);
            }
        }
        if (source.dense() && destination.dense())
            CUDA(cudaMemcpyAsync(destination.data, input, source.size(), cudaMemcpyDefault, stream));
        else
            CUDA(cudaMemcpy2DAsync(destination.data, size_t(destination.ld()) * bytes(destination.type),
                input, pitch, size_t(source.cols) * bytes(source.type), source.rows, cudaMemcpyDefault, stream));
        if (!non_blocking && (source.device < 0 || destination.device < 0)) sync();
    }
    void zero(const Buffer &buffer) {
        DeviceScope scope(device);
        CUDA(cudaMemsetAsync(buffer.data(), 0, buffer.size(), stream));
    }
    void upload(void *dst, const void *src, size_t n) {
        if (!n) return;
        if (transfer_base_) {
            void *snapshot = transfer(n);
            std::memcpy(snapshot, src, n);
            src = snapshot;
        }
        CUDA(cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, stream));
    }
    void download(void *dst, const void *src, size_t n) {
        while (n) {
            size_t count = transfer_base_ ? std::min(n, transfer_bytes_) : n;
            void *staging = transfer_base_ ? transfer(count) : dst;
            CUDA(cudaMemcpyAsync(staging, src, count, cudaMemcpyDeviceToHost, stream));
            sync();
            if (staging != dst) std::memcpy(dst, staging, count);
            dst = static_cast<char *>(dst) + count;
            src = static_cast<const char *>(src) + count;
            n -= count;
        }
    }
    // Called once after loading, planning and capture. Every subsequent Buffer
    // allocation borrows fixed backing, including eager activations and KV chunks.
    // Zero seals an entirely preallocated runtime against any further allocation.
    void reserve_runtime(size_t size) {
        if (runtime_)
            throw std::logic_error("invalid runtime reservation");
        size = Tensor::aligned(size);
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
        if (runtime_) {
            try {
                return static_cast<char *>(runtime_base_) + runtime_->allocate(Tensor::aligned(size));
            } catch (const std::bad_alloc &) {
                throw DeviceOutOfMemory("initialized runtime pool exhausted: needs " +
                                        std::to_string(size) + " bytes, largest free region " +
                                        std::to_string(runtime_->largest()) + " bytes");
            }
        }
        DeviceScope scope(device);
        cudaStreamCaptureStatus status;
        CUDA(cudaStreamIsCapturing(stream, &status));
        if (status != cudaStreamCaptureStatusNone)
            throw std::logic_error("device allocation during graph capture");
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
        transfer_used_ = 0;
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
    size_t runtime_capacity_bytes() const { return runtime_bytes_; }
    bool owns_runtime(const void *data) const {
        return runtime_ && uintptr_t(data) - uintptr_t(runtime_base_) < runtime_bytes_;
    }
    bool can_allocate(std::initializer_list<size_t> sizes) const {
        if (!runtime_)
            throw std::logic_error("runtime pool has not been initialized");
        auto planned = *runtime_;
        try {
            for (auto size : sizes)
                if (size)
                    planned.allocate(Tensor::aligned(size));
            return true;
        } catch (const std::bad_alloc &) {
            return false;
        }
    }
};
// Bump allocations borrow a stable block from the shared pool.
// All graph invalidation happens in ExecutionResources before this block grows.
class DeviceArena : public ScratchAllocator {
    DeviceContext &memory_;
    Buffer storage_;
    size_t used_ = 0;

  public:
    explicit DeviceArena(DeviceContext &memory, size_t capacity = 0) : memory_(memory) {
        reserve(capacity);
    }
    DeviceArena(const DeviceArena &) = delete;
    int device_ordinal() const override { return memory_.device; }
    void reserve(size_t n) {
        if (n <= capacity())
            return;
        if (used_)
            throw std::logic_error("cannot grow an active device arena");
        memory_.sync();
        storage_ = Buffer();
        Tensor::Scope allocation(memory_);
        storage_ = Buffer(Tensor::aligned(n));
        memory_.sync();
    }
    void *allocate(size_t size) override {
        size_t n = Tensor::aligned(size);
        if (n > capacity() - used_)
            throw std::runtime_error("reserved device workspace exhausted");
        auto *base = static_cast<char *>(storage_.data());
        void *p = base ? base + used_ : nullptr;
        used_ += n;
        return p;
    }
    void reset() {
        used_ = 0;
    }
    size_t position() const noexcept override { return used_; }
    void rewind(size_t position) noexcept override {
        if (position > used_) std::terminate();
        used_ = position;
    }
    size_t capacity() const {
        return storage_.size();
    }
};
} // namespace tts
