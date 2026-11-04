#include "core/tensor_storage.h"
#include "core/allocator.h"
#include <utility>
#include "runtime/cuda_context.h"
#include <cstdlib>
#include <cstring>

namespace tts {
namespace {
class CPUAllocator final : public BufferAllocator {
  public:
    void *allocate(size_t bytes) override {
        if (void *data = std::malloc(bytes)) return data;
        throw std::bad_alloc();
    }
    void deallocate(void *data) noexcept override { std::free(data); }
};
CPUAllocator &cpu_allocator() {
    static CPUAllocator allocator;
    return allocator;
}

thread_local BufferAllocator *active_allocator = nullptr;
BufferAllocator &current_allocator() {
    return active_allocator ? *active_allocator : execution_context().memory;
}
BufferAllocator &allocator_for(Device device) {
    if (device.is_cpu()) return cpu_allocator();
    if (active_allocator && active_allocator->device_ordinal() == device.index()) return *active_allocator;
    return execution_context(device.index()).memory;
}
} // namespace

Tensor::Storage::Storage(size_t bytes, BufferAllocator &allocator)
    : size_(bytes), allocator_(&allocator), owner_(allocator.lifetime()), device_(allocator.device_ordinal()) {
    if (bytes) data_ = allocator.allocate(bytes);
}
Tensor::Storage::Storage(size_t bytes) : Storage(bytes, current_allocator()) {}
Tensor::Storage::Storage(size_t bytes, Device device) : Storage(bytes, allocator_for(device)) {}
Tensor::Storage::~Storage() { if (data_) allocator_->deallocate(data_); }
Tensor::Storage::Storage(Storage &&other) noexcept
    : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)),
      allocator_(other.allocator_), owner_(std::move(other.owner_)), device_(other.device_) {}
Tensor::Storage &Tensor::Storage::operator=(Storage &&other) noexcept {
    if (this != &other) {
        if (data_) allocator_->deallocate(data_);
        data_ = std::exchange(other.data_, nullptr);
        size_ = std::exchange(other.size_, 0);
        allocator_ = other.allocator_;
        owner_ = std::move(other.owner_);
        device_ = other.device_;
    }
    return *this;
}
TensorView Tensor::Storage::view(int rows, int columns, Type dtype, size_t offset, int stride) const {
    if (offset > size_ || offset % bytes(dtype)) throw std::out_of_range("tensor storage offset");
    auto result = TensorView::metadata(rows, columns, dtype, stride, device_);
    if (result.span_bytes() > size_ - offset) throw std::out_of_range("tensor view exceeds storage");
    if (data_) result.data = static_cast<char *>(data_) + offset;
    return result;
}
Tensor::Scope::Scope() {
    scratch_ = dynamic_cast<ScratchAllocator *>(&current_allocator());
    if (scratch_) position_ = scratch_->position();
}
Tensor::Scope::Scope(BufferAllocator &allocator) : previous_(active_allocator), binding_(true) {
    active_allocator = &allocator;
    scratch_ = dynamic_cast<ScratchAllocator *>(&allocator);
    if (scratch_) position_ = scratch_->position();
}
Tensor::Scope::~Scope() {
    if (scratch_) scratch_->rewind(position_);
    if (binding_) active_allocator = previous_;
}
TensorView Tensor::scratch(TensorShape shape, Type dtype) {
    auto &allocator = current_allocator();
    auto result = TensorView::metadata(shape.rows, shape.cols, dtype, 0, allocator.device_ordinal());
    result.data = allocator.allocate(checked_size(shape, dtype));
    return result;
}
void Tensor::reserve(size_t bytes) { current_allocator().allocate(bytes); }
size_t Tensor::plan(const std::function<void()> &function) {
    WorkspacePlan planning;
    Scope scope(planning);
    function();
    return planning.size;
}
BufferAllocator *Tensor::resource(const Storage &storage) {
    return storage.owner_ ? storage.owner_.get() : storage.allocator_;
}
Tensor::Tensor(TensorShape shape, Type dtype, BufferAllocator &allocator)
    : storage_(std::make_shared<Storage>(checked_size(shape, dtype), allocator)),
      tensor_(storage_->view(shape.rows, shape.cols, dtype)) {}
Tensor::Tensor(TensorShape shape, Type dtype, Device device)
    : Tensor(shape, dtype, allocator_for(device)) {}

Tensor::Tensor(TensorView source) : Tensor({source.rows, source.cols}, source.type) {
    if (source.device >= 0 || !source.dense() || (source.size() && !source.data))
        throw std::invalid_argument("host tensor construction requires dense CPU data");
    if (source.size()) std::memcpy(data_ptr(), source.data, source.size());
}

Tensor Tensor::view(int rows, int columns) const {
    if (rows == -1 || columns == -1) {
        const int known = rows == -1 ? columns : rows;
        if (known < 1 || numel() % known || numel() / known > size_t(INT32_MAX))
            throw std::invalid_argument("cannot infer tensor view dimension");
        const int inferred = int(numel() / known);
        if (rows == -1) rows = inferred;
        else columns = inferred;
    }
    return view(TensorShape{rows, columns});
}

Tensor Tensor::zeros(TensorShape shape, Type dtype, Device device) {
    Tensor result(shape, dtype, device);
    if (device.is_cpu()) {
        if (result.nbytes()) std::memset(result.data_ptr(), 0, result.nbytes());
    } else execution_context(device.index()).memory.zero(*result.storage_);
    return result;
}

Tensor Tensor::to(Device destination, bool non_blocking) const {
    if (!defined()) throw std::logic_error("cannot transfer an undefined tensor");
    const auto source = device();
    if (source == destination) return *this;
    if (!is_contiguous())
        throw std::invalid_argument("strided transfer requires an explicit contiguous()");
    if (source.is_cuda() && destination.is_cuda())
        throw std::invalid_argument("cross-device CUDA tensor transfers are not supported");
    auto *context = dynamic_cast<DeviceContext *>(resource(*storage_));
    const int ordinal = source.is_cuda() ? source.index() : destination.index();
    if (!context || context->device != ordinal) context = &execution_context(ordinal).memory;
    BufferAllocator *allocator = context;
    if (destination.is_cpu()) {
        if (non_blocking || context->has_host_tensor_backing()) allocator = &context->host_allocator();
        else allocator = &cpu_allocator();
    }
    Tensor result(shape(), dtype(), *allocator);
    context->copy(view(), result.view(), non_blocking);
    return result;
}

Tensor Tensor::clone() const {
    if (!defined()) throw std::logic_error("cannot clone an undefined tensor");
    BufferAllocator *allocator = &cpu_allocator();
    if (device().is_cuda()) {
        auto *context = dynamic_cast<DeviceContext *>(resource(*storage_));
        allocator = context ? context : &execution_context(device().index()).memory;
    }
    Tensor result(shape(), dtype(), *allocator);
    result.copy_(*this, device().is_cuda());
    return result;
}

Tensor &Tensor::copy_(const Tensor &source, bool non_blocking) {
    if (!defined() || !source.defined() || dtype() != source.dtype() ||
        shape().rows != source.shape().rows || shape().cols != source.shape().cols)
        throw std::invalid_argument("tensor copy requires matching shape and dtype");
    if (!nbytes()) return *this;
    if (view().data == source.view().data && stride(0) == source.stride(0)) return *this;
    if (device() == source.device() && view().overlaps(source.view()))
        throw std::invalid_argument("overlapping tensor copy requires an explicit clone");
    if (device().is_cpu() && source.device().is_cpu()) {
        source.synchronize();
        synchronize();
        for (int row = 0; row < size(0); ++row)
            std::memcpy(static_cast<char *>(data_ptr()) + size_t(row) * stride(0) * element_size(),
                static_cast<const char *>(source.data_ptr()) + size_t(row) * source.stride(0) * element_size(),
                size_t(size(1)) * element_size());
    } else {
        const int ordinal = device().is_cuda() ? device().index() : source.device().index();
        auto *context = dynamic_cast<DeviceContext *>(resource(*storage_));
        if (!context) context = dynamic_cast<DeviceContext *>(resource(*source.storage_));
        if (!context || context->device != ordinal) context = &execution_context(ordinal).memory;
        context->copy(source.view(), view(), non_blocking);
    }
    return *this;
}

void Tensor::synchronize() const {
    if (!defined()) return;
    if (auto *context = dynamic_cast<DeviceContext *>(resource(*storage_))) context->sync();
    else if (device().is_cuda()) execution_context(device().index()).memory.sync();
}
} // namespace tts
