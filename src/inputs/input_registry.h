#pragma once
#include "execution/request_planner.h"
#include "core/tensor.h"

namespace tts {
class ModelLoader;
std::unique_ptr<RequestPlanner> load_inputs(const ModelLoader &, int batch, int context, Type);
} // namespace tts
