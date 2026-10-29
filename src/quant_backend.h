#pragma once
#include "device.h"
#include "quantization.h"
#include "kernel_problem.h"
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
    QuantCacheStats cache_stats() const;
    uint64_t execution_revision() const;
    void prepare(int rows, int output_channels, int input_channels);
    void prepare_gated(int rows, int width, int kind);
    void prepare_norm(int rows, int width, bool gemma);
    void prepare_projection(int rows, int output_channels, int input_channels);
    void quantize(TensorView source, RowScaledView destination);
    void linear(RowScaledView input, RowScaledView weight, TensorView output);
    void project(TensorView input, RowScaledView weight, RowScaledView scratch, TensorView output);
    void norm(TensorView residual, TensorView update, TensorView weight, RowScaledView output,
              float eps, bool gemma, TensorView update_scale = {});
    void gated(TensorView gate_up, RowScaledView output, int kind);
};
} // namespace tts
