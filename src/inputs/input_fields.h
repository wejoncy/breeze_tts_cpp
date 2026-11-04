#pragma once
#include "core/input_batch.h"
#include "core/host_tensor.h"
#include "core/sequence_builder.h"

namespace tts {
inline TensorView own_tensor(InputBatch &input, TensorShape shape, Type type) {
    auto storage = std::make_shared<HostTensor>(shape, type);
    auto result = storage->view();
    input.owners.push_back(std::move(storage));
    return result;
}
inline TensorView own_integers(InputBatch &input, const std::vector<int> &values, int columns = 1) {
    if (columns < 1 || values.size() % size_t(columns) || values.size() / columns > INT32_MAX)
        throw std::invalid_argument("input integer tensor shape");
    auto result = own_tensor(input, {int(values.size() / columns), columns}, Type::I32);
    std::copy(values.begin(), values.end(), result.ptr<int>());
    return result;
}
inline InputSequence own_sequence(InputBatch &input, const std::vector<int> &lengths) {
    auto values = SequenceBuilder::pack(lengths);
    int rows = 0, maximum = 0;
    for (int length : lengths) {
        if (length < 1 || length > INT32_MAX - rows) throw std::overflow_error("input sequence rows");
        rows += length;
        maximum = std::max(maximum, length);
    }
    return {own_integers(input, values), {int(lengths.size()), maximum, rows}};
}
} // namespace tts
