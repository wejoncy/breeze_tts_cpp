#include "core/input_batch.h"
#include "core/sequence_builder.h"
#include <cstring>
#include <map>

namespace tts {
namespace {
const InputSequence &find_sequence(const InputBatch &input, std::string_view name) {
    if (name.empty()) return input.sequence;
    for (const auto &sequence : input.sequences)
        if (sequence.name == name) return sequence.value;
    throw std::invalid_argument("input offset refers to an unknown sequence");
}
InputSequence pack_sequences(ArrayView<const InputBatch *> inputs, std::string_view name,
                              InputBatch &owner) {
    SequenceBuilder builder;
    int64_t total_rows = 0, total_sequences = 0;
    for (size_t input_index = 0; input_index < inputs.size; ++input_index) {
        const auto shape = find_sequence(*inputs[input_index], name).shape;
        if (shape.rows < 0 || shape.batch < 0 || shape.time < 0)
            throw std::invalid_argument("invalid input sequence dimensions");
        total_rows += shape.rows;
        total_sequences += shape.batch;
        if (2 * total_rows + 2 * total_sequences + 1 > INT32_MAX)
            throw std::overflow_error("packed sequence metadata capacity");
    }
    builder.reserve(int(total_sequences), int(total_rows));
    for (size_t input_index = 0; input_index < inputs.size; ++input_index) {
        const auto &source = find_sequence(*inputs[input_index], name);
        const auto shape = source.shape;
        if (source.metadata.numel()) {
            if (source.metadata.device != -1 || source.metadata.type != Type::I32 ||
                !source.metadata.dense() || !source.metadata.data ||
                source.metadata.numel() != size_t(2) * shape.rows + size_t(2) * shape.batch + 1)
                throw std::invalid_argument("invalid host sequence metadata");
            const auto *offsets = source.metadata.ptr<int>() + size_t(shape.rows) * 2;
            const auto *lengths = offsets + shape.batch + 1;
            if (offsets[0] || offsets[shape.batch] != shape.rows)
                throw std::invalid_argument("sequence offsets do not cover their tensor rows");
            for (int sequence = 0; sequence < shape.batch; ++sequence)
                builder.append(offsets[sequence + 1] - offsets[sequence], lengths[sequence]);
        } else {
            if (shape.rows != int64_t(shape.batch) * shape.time)
                throw std::invalid_argument("uniform sequence row count mismatch");
            for (int sequence = 0; sequence < shape.batch; ++sequence)
                builder.append(shape.time, shape.time);
        }
    }
    const auto layout = builder.view();
    auto storage = std::make_shared<std::vector<int>>(std::move(builder).release());
    TensorView metadata(storage->data(), int(storage->size()), 1, Type::I32);
    owner.owners.push_back(storage);
    return {metadata, {layout.sequences, layout.max_length, layout.rows}};
}
} // namespace
InputBatch merge_inputs(ArrayView<const InputBatch *> inputs) {
    if (!inputs.size) throw std::invalid_argument("cannot merge an empty input batch");
    InputBatch result;
    result.fixed_position = inputs[0]->fixed_position;
    result.offsets = inputs[0]->offsets;
    result.batch_bindings = inputs[0]->batch_bindings;
    result.parameters = inputs[0]->parameters;
    result.sequence = pack_sequences(inputs, {}, result);
    for (const auto &sequence : inputs[0]->sequences)
        result.sequences.push_back({sequence.name, pack_sequences(inputs, sequence.name, result)});
    auto fields = [&](bool upload) {
        auto &destination = upload ? result.uploads : result.bindings;
        const auto &prototype = upload ? inputs[0]->uploads : inputs[0]->bindings;
        for (const auto &field : prototype) {
            std::vector<const TensorView *> sources;
            sources.reserve(inputs.size);
            int64_t rows = 0;
            for (size_t input_index = 0; input_index < inputs.size; ++input_index) {
                const auto &source = upload ? inputs[input_index]->uploads : inputs[input_index]->bindings;
                auto found = std::find_if(source.begin(), source.end(),
                    [&](const auto &entry) { return entry.name == field.name; });
                if (found == source.end() || found->value.device != -1 || !found->value.dense() ||
                    found->value.cols != field.value.cols || found->value.type != field.value.type ||
                    (found->value.numel() && !found->value.data))
                    throw std::invalid_argument("input tensor layouts do not match for packing");
                rows += found->value.rows;
                if (rows > INT32_MAX) throw std::overflow_error("packed tensor rows");
                sources.push_back(&found->value);
            }
            auto storage = std::make_shared<std::vector<char>>(size_t(rows) * field.value.cols * bytes(field.value.type));
            size_t byte_offset = 0;
            std::map<std::string_view, int64_t> bases;
            for (const auto &offset : result.offsets)
                if (offset.tensor == field.name) bases.try_emplace(offset.sequence, 0);
            for (size_t input_index = 0; input_index < inputs.size; ++input_index) {
                const auto &source = *sources[input_index];
                if (source.size()) std::memcpy(storage->data() + byte_offset, source.data, source.size());
                for (const auto &offset : result.offsets) {
                    if (offset.tensor != field.name) continue;
                    if (source.type != Type::I32 || offset.column < 0 || offset.column >= source.cols)
                        throw std::invalid_argument("input relocation requires an I32 tensor column");
                    auto *values = reinterpret_cast<int *>(storage->data() + byte_offset);
                    for (int row = 0; row < source.rows; ++row) {
                        auto &value = values[size_t(row) * source.cols + offset.column];
                        if (value >= 0) {
                            const int64_t shifted = value + bases[offset.sequence];
                            if (shifted > INT32_MAX) throw std::overflow_error("packed input row offset");
                            value = int(shifted);
                        }
                    }
                }
                for (auto &[axis, base] : bases)
                    base += find_sequence(*inputs[input_index], axis).shape.rows;
                byte_offset += source.size();
            }
            destination.push_back({field.name, TensorView(storage->data(), int(rows), field.value.cols, field.value.type)});
            result.owners.push_back(storage);
        }
    };
    fields(true);
    fields(false);
    int primary_base = 0;
    for (size_t input_index = 0; input_index < inputs.size; ++input_index) {
        const auto &input = *inputs[input_index];
        if (input.fixed_position != result.fixed_position)
            throw std::invalid_argument("packed inputs disagree on the position base");
        for (auto region : input.regions) {
            region.first_row += primary_base;
            result.regions.push_back(region);
        }
        primary_base += input.sequence.shape.rows;
        result.limits.batch += input.limits.batch;
        result.limits.time = std::max(result.limits.time, input.limits.time);
        result.limits.rows += input.limits.rows;
        result.owners.insert(result.owners.end(), input.owners.begin(), input.owners.end());
    }
    return result;
}
} // namespace tts
