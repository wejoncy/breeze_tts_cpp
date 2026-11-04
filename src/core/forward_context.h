#pragma once
#include "core/forward_batch.h"
#include "core/buffer.h"
#include <cstddef>
#include <initializer_list>
#include <string>
#include <string_view>

namespace tts {
template <class T> struct ArrayView {
    T *data = nullptr;
    size_t size = 0;
    T &operator[](size_t index) const {
        if (index >= size) throw std::out_of_range("forward argument index");
        return data[index];
    }
};
template <class Value> struct NamedValue {
    std::string_view name;
    Value value;
};
template <class Item> struct NamedView : ArrayView<Item> {
    NamedView() = default;
    NamedView(Item *arguments, size_t count) : ArrayView<Item>{arguments, count} {}
    bool contains(std::string_view name) const {
        for (size_t index = 0; index < this->size; ++index)
            if (this->data[index].name == name) return true;
        return false;
    }
    auto &operator[](std::string_view name) const {
        for (size_t argument_index = 0; argument_index < this->size; ++argument_index)
            if (this->data[argument_index].name == name)
                return this->data[argument_index].value;
        throw std::out_of_range("missing forward argument: " + std::string(name));
    }
};
using NamedTensor = NamedValue<TensorView>;
using TensorInputs = NamedView<const NamedTensor>;
using TensorOutputs = NamedView<NamedTensor>;
struct TensorRegion {
    TensorView tensor;
    int first_row;
};
struct ForwardShape {
    int batch = 0, time = 0, rows = 0;
};
struct ForwardCall {
    const void *owner;
    const char *name;
    std::initializer_list<int> shape;
    std::initializer_list<const void *> buffers;
    const void *arguments;
    void (*invoke)(const void *);
};

// Borrowed computation data. Resource ownership and request state stay in the runner.
struct ForwardContext {
    enum Mode { Plan, Prepare, Capture, Execute } mode = Execute;
    ForwardBatch batch;
    NamedView<const NamedValue<SequenceLayout>> sequences;
    NamedView<const NamedValue<double>> parameters;
    NamedView<const NamedValue<ArrayView<const TensorView>>> state;
    ArrayView<const TensorRegion> regions;
    ArrayView<const int> batches;
    bool allow_replay = true;
    void *executor = nullptr;
    bool (*dispatch)(void *, const ForwardCall &, bool) = nullptr;
    int max_batch, max_time, max_rows;
    ForwardContext(int batches, int time, int rows)
        : max_batch(batches), max_time(time), max_rows(rows) {}
    template <class Function>
    void run(const void *owner, const char *name, std::initializer_list<int> shape,
             std::initializer_list<const void *> buffers, const Function &function) const {
        if (mode != Execute && mode != Capture) return;
        ForwardCall call{owner, name, shape, buffers, &function,
            [](const void *arguments) { (*static_cast<const Function *>(arguments))(); }};
        if (dispatch && (allow_replay || mode == Capture) && dispatch(executor, call, mode == Capture)) return;
        if (mode == Execute) function();
        else if (mode == Capture) throw std::logic_error("forward capture has no executor");
    }
    TensorView tensor(TensorShape shape, Type type) const {
        return Tensor::scratch(shape, type);
    }
};
} // namespace tts
