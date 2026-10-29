#pragma once
#include "device.h"
#include "quantization.h"
#include <memory>
namespace tts {
class QuantNorm {
    struct Impl;
    std::unique_ptr<Impl> impl;

  public:
    QuantNorm(DeviceContext &memory, Type format = Type::FP8);
    ~QuantNorm();
    void begin_admission();
    uint64_t execution_revision() const;
    void prepare(int rows, int width, bool gemma);
    void run(TensorView residual, TensorView update, TensorView weight, RowScaledView output,
             float eps, bool gemma, TensorView update_scale);
};
} // namespace tts
