#pragma once
#include "device.h"
#include "quantization.h"
#include <memory>
namespace tts {
class QuantGated {
    struct Impl;
    std::unique_ptr<Impl> impl;

  public:
    QuantGated(DeviceContext &memory, Type format = Type::FP8);
    ~QuantGated();
    void begin_admission();
    uint64_t execution_revision() const;
    void prepare(int rows, int width, int kind);
    void run(TensorView input, RowScaledView output, int kind);
};
} // namespace tts
