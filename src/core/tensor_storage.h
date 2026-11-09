#pragma once
#include "core/tensor.h"
#include "core/device.h"
#include "core/allocator.h"
#include <memory>
#include <utility>
namespace tts {
class BufferAllocator;
class ScratchAllocator;
// Copies share storage; views borrow it. Data movement is explicit through to().
class Tensor {
  public:
    class Storage {
        friend class Tensor;
        void *data_ = nullptr;
        size_t size_ = 0;
        BufferAllocator *allocator_ = nullptr;
        std::shared_ptr<BufferAllocator> owner_;
        int device_ = -1;
      public:
        Storage() = default;
        Storage(size_t bytes, BufferAllocator &);
        explicit Storage(size_t bytes);
        Storage(size_t bytes, Device device);
        ~Storage();
        Storage(const Storage &) = delete;
        Storage &operator=(const Storage &) = delete;
        Storage(Storage &&) noexcept;
        Storage &operator=(Storage &&) noexcept;
        void *data() const { return data_; }
        size_t size() const { return size_; }
        int device() const { return device_; }
        template <class Value> Value *ptr() const { return static_cast<Value *>(data_); }
        TensorView view(int rows, int columns, Type dtype = Type::BF16, size_t offset = 0, int stride = 0) const;
        Storage &copy_(TensorView source, bool non_blocking = false);
    };
    class Scope {
        BufferAllocator *previous_ = nullptr;
        ScratchAllocator *scratch_ = nullptr;
        size_t position_ = 0;
        bool binding_ = false;
      public:
        Scope();
        explicit Scope(BufferAllocator &);
        ~Scope();
        Scope(const Scope &) = delete;
        Scope &operator=(const Scope &) = delete;
    };

  private:
    std::shared_ptr<Storage> storage_;
    TensorView tensor_;
    Tensor(std::shared_ptr<Storage> storage, TensorView view)
        : storage_(std::move(storage)), tensor_(view) {}
    Tensor(TensorShape, Type, BufferAllocator &);
    static BufferAllocator *resource(const Storage &);

  public:
    Tensor() = default;
    explicit Tensor(TensorShape shape, Type dtype = Type::F32, Device device = Device::cpu());
    explicit Tensor(TensorView source);
    Tensor(Storage storage, TensorShape shape, Type dtype)
        : storage_(std::make_shared<Storage>(std::move(storage))),
          tensor_(storage_->view(shape.rows, shape.cols, dtype)) {}
    static Tensor empty(TensorShape shape, Type dtype = Type::F32, Device device = Device::cpu()) {
        return Tensor(shape, dtype, device);
    }
    static Tensor zeros(TensorShape shape, Type dtype = Type::F32, Device device = Device::cpu());
    static TensorView scratch(TensorShape, Type);
    static void reserve(size_t bytes);
    template <class Forward> static size_t plan(Forward &&forward) {
        WorkspacePlan planning;
        Scope scope(planning);
        forward();
        return planning.size;
    }
    static size_t aligned(size_t bytes) {
        return aligned_storage_bytes(bytes);
    }
    bool defined() const { return bool(storage_); }
    const TensorShape &shape() const { return tensor_; }
    Type dtype() const { return tensor_.type; }
    Device device() const { return Device(tensor_.device); }
    size_t numel() const { return tensor_.numel(); }
    size_t element_size() const { return bytes(dtype()); }
    size_t nbytes() const { return tensor_.size(); }
    int ndim() const { return 2; }
    bool is_contiguous() const { return tensor_.dense(); }
    int size(int dimension) const {
        if (dimension < 0) dimension += ndim();
        if (dimension == 0) return tensor_.rows;
        if (dimension == 1) return tensor_.cols;
        throw std::out_of_range("tensor dimension");
    }
    int stride(int dimension) const {
        if (dimension < 0) dimension += ndim();
        if (dimension == 0) return tensor_.ld();
        if (dimension == 1) return 1;
        throw std::out_of_range("tensor dimension");
    }
    template <class Value = void> Value *data_ptr() const {
        return tensor_.ptr<Value>();
    }
    TensorView view() const { return tensor_; }
    Tensor view(TensorShape shape) const { return Tensor(storage_, tensor_.reshape(shape)); }
    Tensor view(int rows, int columns) const;
    Tensor reshape(TensorShape shape) const { return view(shape); }
    Tensor reshape(int rows, int columns) const { return view(rows, columns); }
    Tensor narrow(int dimension, int first, int count) const {
        if (dimension < 0) dimension += ndim();
        if (dimension == 0) return Tensor(storage_, tensor_.slice(first, count));
        if (dimension == 1) return Tensor(storage_, tensor_.columns(first, count));
        throw std::out_of_range("tensor dimension");
    }
    Tensor to(Device device, bool non_blocking = false) const;
    Tensor to(std::string_view device, bool non_blocking = false) const {
        return to(Device(device), non_blocking);
    }
    Tensor cpu(bool non_blocking = false) const { return to(Device::cpu(), non_blocking); }
    Tensor cuda(int ordinal = 0, bool non_blocking = false) const {
        return to(Device::cuda(ordinal), non_blocking);
    }
    Tensor clone() const;
    Tensor contiguous() const {
        return is_contiguous() ? *this : clone();
    }
    Tensor &copy_(const Tensor &source, bool non_blocking = false);
    void synchronize() const;

  private:
    static size_t checked_size(TensorShape shape, Type dtype) {
        if (shape.numel() > SIZE_MAX / bytes(dtype))
            throw std::overflow_error("tensor storage size");
        return shape.numel() * bytes(dtype);
    }
};
} // namespace tts
