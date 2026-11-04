#pragma once
#include "runtime/device.h"
#include "core/quantization.h"
#include <memory>
namespace tts {
class QuantNorm {
    struct Impl;
    std::unique_ptr<Impl> impl;

  public:
    QuantNorm(DeviceContext &memory, Type format = Type::FP8);
    ~QuantNorm();
    void run(TensorView residual, TensorView update, TensorView weight, RowScaledView output,
             float eps, bool gemma, TensorView update_scale);
};
} // namespace tts
