#pragma once
#include "runtime/device.h"
#include "core/quantization.h"
#include <memory>
namespace tts {
class QuantGated {
    struct Impl;
    std::unique_ptr<Impl> impl;

  public:
    QuantGated(DeviceContext &memory, Type format = Type::FP8);
    ~QuantGated();
    void run(TensorView input, RowScaledView output, int kind);
};
} // namespace tts
