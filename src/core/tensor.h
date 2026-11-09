#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace tts {
enum class Type { BF16, F32, I32, FP8, I8, F16, I16, I64, F64 };
inline size_t bytes(Type type) {
    if (type == Type::I64 || type == Type::F64) return 8;
    return (type == Type::FP8 || type == Type::I8) ? 1
        : (type == Type::BF16 || type == Type::F16 || type == Type::I16) ? 2 : 4;
}
// Operators flatten batch/time into rows. Only the innermost axis must be dense.
// This metadata layer has no CUDA, allocator, model, or ABI dependencies.
struct TensorShape {
    int rows = 0, cols = 0;
    TensorShape() = default;
    TensorShape(int row_count, int column_count) : rows(row_count), cols(column_count) {
        if (row_count < 0 || column_count < 0)
            throw std::invalid_argument("negative tensor dimension");
    }
    size_t numel() const {
        return size_t(rows) * size_t(cols);
    }
};
struct TensorView : TensorShape {
    void *data = nullptr;
    Type type = Type::BF16;
    int stride = 0; // row stride in elements; zero means dense
    int device = -1; // CUDA ordinal; -1 is host or metadata-only storage.
    TensorView() = default;
    TensorView(void *pointer, int row_count, int column_count, Type data_type = Type::BF16,
               int leading_dimension = 0, int ordinal = -1)
        : TensorShape(row_count, column_count), data(pointer), type(data_type),
          stride(leading_dimension), device(ordinal) {
        validate_layout();
        if (numel() && !pointer)
            throw std::invalid_argument("invalid tensor pointer/stride");
    }
    static TensorView metadata(int rows, int cols, Type type, int stride = 0, int device = -1) {
        TensorView result;
        static_cast<TensorShape &>(result) = TensorShape(rows, cols);
        result.type = type; result.stride = stride; result.device = device;
        result.validate_layout();
        return result;
    }
    int ld() const {
        return stride ? stride : cols;
    }
    bool dense() const {
        return rows <= 1 || ld() == cols;
    }
    size_t size() const {
        return numel() * bytes(type);
    }
    size_t span_elements() const {
        return rows && cols ? size_t(rows - 1) * ld() + cols : 0;
    }
    size_t span_bytes() const {
        return span_elements() * bytes(type);
    }
    bool overlaps(TensorView other) const {
        if (!data || !other.data) return false;
        auto first = reinterpret_cast<uintptr_t>(data);
        auto second = reinterpret_cast<uintptr_t>(other.data);
        return first < second + other.span_bytes() && second < first + span_bytes();
    }
    TensorView slice(int row, int n = 1) const {
        if (row < 0 || n < 0 || row > rows || n > rows - row)
            throw std::out_of_range("tensor row slice");
        return alias(offset(size_t(row) * ld()), n, cols, stride);
    }
    TensorView columns(int first, int count) const {
        if (first < 0 || count < 0 || first > cols || count > cols - first)
            throw std::out_of_range("tensor column slice");
        return alias(offset(first), rows, count, ld());
    }
    TensorView every_rows(int first, int step, int count) const {
        if (first < 0 || step < 1 || count < 0 || first > rows ||
            (count && (first >= rows || size_t(count - 1) * step >= size_t(rows - first))) ||
            size_t(ld()) * step > size_t(INT32_MAX))
            throw std::out_of_range("strided tensor rows");
        return alias(offset(size_t(first) * ld()), count, cols, ld() * step);
    }
    TensorView reshape(TensorShape shape) const {
        if (!dense() || shape.numel() != numel())
            throw std::invalid_argument("reshape requires dense storage and equal element count");
        return alias(data, shape.rows, shape.cols, 0);
    }
    template <typename T> T *ptr() const {
        return static_cast<T *>(data);
    }
    // Explicit transfer into existing storage; the owner retains both views
    // until completion when non_blocking is true.
    const TensorView &copy_(TensorView source, bool non_blocking = false) const;

  private:
    void validate_layout() const {
        if (stride < 0 || (stride && stride < cols))
            throw std::invalid_argument("invalid tensor stride");
        if (span_elements() > std::numeric_limits<size_t>::max() / bytes(type))
            throw std::overflow_error("tensor byte size overflow");
    }
    TensorView alias(void *pointer, int row_count, int column_count, int leading) const {
        auto result = metadata(row_count, column_count, type, leading, device);
        result.data = pointer;
        return result;
    }
    void *offset(size_t elements) const {
        return data ? static_cast<char *>(data) + elements * bytes(type) : nullptr;
    }
};
} // namespace tts
