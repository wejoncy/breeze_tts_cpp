#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace tts {
enum class Type { BF16, F32, I32, FP8, I8 };
inline size_t bytes(Type t) {
    return (t == Type::FP8 || t == Type::I8) ? 1 : t == Type::BF16 ? 2 : 4;
}
// Operators flatten batch/time into rows. Only the innermost axis must be dense.
// This metadata layer has no CUDA, allocator, model, or ABI dependencies.
struct TensorShape {
    int rows = 0, cols = 0;
    TensorShape() = default;
    TensorShape(int r, int c) : rows(r), cols(c) {
        if (r < 0 || c < 0)
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
    TensorView() = default;
    TensorView(void *p, int r, int c, Type t = Type::BF16, int s = 0)
        : TensorShape(r, c), data(p), type(t), stride(s) {
        if (s < 0 || (s && s < c) || (numel() && !p))
            throw std::invalid_argument("invalid tensor pointer/stride");
        if (span_elements() > std::numeric_limits<size_t>::max() / bytes(t))
            throw std::overflow_error("tensor byte size overflow");
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
    TensorView slice(int row, int n = 1) const {
        if (row < 0 || n < 0 || row > rows || n > rows - row)
            throw std::out_of_range("tensor row slice");
        return {offset(size_t(row) * ld()), n, cols, type, stride};
    }
    TensorView columns(int first, int count) const {
        if (first < 0 || count < 0 || first > cols || count > cols - first)
            throw std::out_of_range("tensor column slice");
        return {offset(first), rows, count, type, ld()};
    }
    TensorView every_rows(int first, int step, int count) const {
        if (first < 0 || step < 1 || count < 0 || first > rows ||
            (count && (first >= rows || size_t(count - 1) * step >= size_t(rows - first))) ||
            size_t(ld()) * step > size_t(INT32_MAX))
            throw std::out_of_range("strided tensor rows");
        return {offset(size_t(first) * ld()), count, cols, type, ld() * step};
    }
    TensorView reshape(TensorShape s) const {
        if (!dense() || s.numel() != numel())
            throw std::invalid_argument("reshape requires dense storage and equal element count");
        return {data, s.rows, s.cols, type};
    }
    template <typename T> T *ptr() const {
        return static_cast<T *>(data);
    }

  private:
    void *offset(size_t elements) const {
        return data ? static_cast<char *>(data) + elements * bytes(type) : nullptr;
    }
};
} // namespace tts
