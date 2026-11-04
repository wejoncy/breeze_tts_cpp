#pragma once
#include "core/input_batch.h"
#include "core/model.h"

namespace tts {
class DeviceContext;
void run_forward(const Model &, std::string_view program, const InputBatch &, TensorInputs bindings,
                 TensorOutputs outputs, ForwardContext &, DeviceContext *transfers = nullptr);
} // namespace tts
