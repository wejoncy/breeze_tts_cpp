#pragma once
#include "core/forward_context.h"

namespace tts {
class Model {
  public:
    virtual ~Model() = default;
    virtual void forward(std::string_view program, TensorInputs inputs, TensorOutputs outputs,
                          ForwardContext &context) const = 0;
};
} // namespace tts
