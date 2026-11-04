#pragma once
#include "core/input_batch.h"
#include "core/model.h"

namespace tts {
void run_forward(const Model &, std::string_view program, const InputBatch &, TensorInputs bindings,
                 TensorOutputs outputs, ForwardContext &);
} // namespace tts
