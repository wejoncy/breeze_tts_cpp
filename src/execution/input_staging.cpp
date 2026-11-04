#include "execution/input_staging.h"
#include "runtime/device.h"

namespace tts {
void run_forward(const Model &model, std::string_view program, const InputBatch &input,
                 TensorInputs bindings, TensorOutputs outputs, ForwardContext &context,
                 DeviceContext *transfers) {
    std::vector<std::pair<TensorView, TensorView>> uploaded;
    auto upload = [&](TensorView source) {
        for (const auto &entry : uploaded)
            if (source.data && source.data == entry.first.data && source.rows == entry.first.rows &&
                source.cols == entry.first.cols && source.stride == entry.first.stride &&
                source.type == entry.first.type)
                return entry.second;
        auto destination = context.tensor({source.rows, source.cols}, source.type);
        if (context.mode == ForwardContext::Execute && source.numel()) {
            if (!transfers || source.device != -1 || !source.dense() || !source.data)
                throw std::invalid_argument("input upload requires dense host data");
            transfers->upload(destination.data, source.data, source.size());
        }
        uploaded.emplace_back(source, destination);
        return destination;
    };
    auto sequence = [&](const InputSequence &source) {
        const auto shape = source.shape;
        if (!source.metadata.numel())
            return SequenceLayout::uniform(shape.batch, shape.time);
        if (source.metadata.type != Type::I32 ||
            source.metadata.numel() != size_t(2) * shape.rows + size_t(2) * shape.batch + 1)
            throw std::invalid_argument("input sequence metadata shape/type mismatch");
        const auto metadata = upload(source.metadata);
        auto pointer = [&](size_t offset) { return metadata.data ? metadata.ptr<int>() + offset : nullptr; };
        return SequenceLayout{pointer(0), pointer(shape.rows), pointer(size_t(shape.rows) * 2),
            pointer(size_t(shape.rows) * 2 + shape.batch + 1), shape.rows, shape.batch, shape.time};
    };
    std::vector<NamedTensor> tensors = input.bindings;
    tensors.reserve(tensors.size() + input.uploads.size() + bindings.size);
    for (const auto &field : input.uploads) tensors.push_back({field.name, upload(field.value)});
    for (const auto &address : input.addresses)
        if (std::none_of(tensors.begin(), tensors.end(),
                [&](const auto &field) { return field.name == address.tensor; })) {
            if (context.mode == ForwardContext::Execute)
                throw std::logic_error("state tensor addresses were not bound before execution");
            if (address.offsets.size() > size_t(INT32_MAX))
                throw std::overflow_error("state tensor address count");
            tensors.push_back({address.tensor,
                context.tensor({int(address.offsets.size()), 1}, Type::I64)});
        }
    for (size_t field = 0; field < bindings.size; ++field) {
        auto found = std::find_if(tensors.begin(), tensors.end(),
            [&](const auto &entry) { return entry.name == bindings.data[field].name; });
        if (found == tensors.end()) tensors.push_back(bindings.data[field]);
        else found->value = bindings.data[field].value;
    }
    std::vector<NamedValue<SequenceLayout>> sequences;
    sequences.reserve(input.sequences.size());
    for (const auto &field : input.sequences) sequences.push_back({field.name, sequence(field.value)});
    std::vector<TensorRegion> regions = input.regions;
    for (auto &region : regions)
        if (region.tensor.device < 0) region.tensor = upload(region.tensor);
    auto invocation = context;
    invocation.batch.sequence = sequence(input.sequence);
    invocation.batch.fixed_position = input.fixed_position;
    invocation.sequences = {sequences.data(), sequences.size()};
    invocation.regions = {regions.data(), regions.size()};
    invocation.parameters = {input.parameters.data(), input.parameters.size()};
    const TensorInputs arguments(tensors.data(), tensors.size());
    for (const auto &binding : input.batch_bindings) {
        const auto tensor = arguments[binding.tensor];
        switch (binding.field) {
        case BatchField::Positions: invocation.batch.positions = tensor.ptr<int>(); break;
        case BatchField::CacheTable: invocation.batch.cache.layout.block_table = tensor.ptr<int>(); break;
        case BatchField::OutputRows: invocation.batch.output_rows = tensor.ptr<int>(); break;
        case BatchField::Bidirectional: invocation.batch.bidirectional = tensor.ptr<AttentionSequence>(); break;
        case BatchField::Causal: invocation.batch.causal = tensor.ptr<AttentionSequence>(); break;
        }
    }
    model.forward(program, arguments, outputs, invocation);
}
} // namespace tts
