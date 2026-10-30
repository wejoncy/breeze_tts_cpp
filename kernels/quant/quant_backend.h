#pragma once
#include "runtime/device.h"
#include "core/quantization.h"
#include "kernels/quant/kernel_problem.h"
#include <memory>

namespace tts {
struct QuantCacheStats {
    SelectionStats selection;
    size_t prepared_plans = 0;
};
class QuantBackend {
    struct Impl;
    std::unique_ptr<Impl> impl;

  public:
    QuantBackend(DeviceContext &memory, Type format = Type::FP8);
    ~QuantBackend();
    QuantBackend(const QuantBackend &) = delete;
    Type type() const;
    void begin_admission();
    void seal();
    QuantCacheStats cache_stats() const;
    void prepare(int rows, int output_channels, int input_channels);
    void quantize(TensorView source, RowScaledView destination);
    void linear(RowScaledView input, RowScaledView weight, TensorView output);
    void norm(TensorView residual, TensorView update, TensorView weight, RowScaledView output,
              float eps, bool gemma, TensorView update_scale = {});
    void gated(TensorView gate_up, RowScaledView output, int kind);
};
} // namespace tts
