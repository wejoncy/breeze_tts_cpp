#pragma once
#include "core/forward_context.h"
#include <memory>
#include <vector>

namespace tts {
// CPU input descriptions borrow validated request data. Uploads and bindings are
// explicit: a binding may already reside on the device or remain host metadata.
struct InputSequence {
    TensorView metadata;
    ForwardShape shape;
};
struct InputOffset {
    std::string_view tensor, sequence;
    int column;
};
enum class BatchField { Positions, CacheTable, OutputRows, Bidirectional, Causal };
struct InputBinding {
    std::string_view tensor;
    BatchField field;
};
struct InputAddress {
    std::string tensor, state;
    std::vector<int64_t> offsets;
};
struct InputBatch {
    ForwardShape limits;
    InputSequence sequence;
    int fixed_position = 0;
    std::vector<NamedTensor> uploads, bindings;
    std::vector<NamedValue<InputSequence>> sequences;
    std::vector<TensorRegion> regions;
    std::vector<InputOffset> offsets;
    std::vector<InputBinding> batch_bindings;
    std::vector<NamedValue<double>> parameters;
    std::vector<InputAddress> addresses;
    std::vector<std::shared_ptr<const void>> owners;
};
InputBatch merge_inputs(ArrayView<const InputBatch *>);
} // namespace tts
