#pragma once
#include "core/buffer.h"
#include "core/quantization.h"
#include "core/kernel_problem.h"
#include <memory>
#include <cublasLt.h>

namespace tts {
namespace ops { struct NormFusion; struct Epilogue; }
struct QkvProjectionOutput;
struct QuantCacheStats {
    SelectionStats selection;
    size_t prepared_plans = 0;
};
class QuantizedGemm {
    struct Impl;
    std::unique_ptr<Impl> impl;

  public:
    QuantizedGemm(cudaStream_t stream, cublasLtHandle_t handle, Buffer &workspace,
                 Type format = Type::FP8);
    ~QuantizedGemm();
    QuantizedGemm(const QuantizedGemm &) = delete;
    void seal();
    QuantCacheStats cache_stats() const;
    // Output dtype is part of the prepared execution contract. Inputs/weights
    // retain quantized storage; encoding belongs to the standalone quantization op.
    void prepare(int rows, int output_channels, int input_channels, Type output_type = Type::BF16,
                 QuantizationGroups groups = {});
    void prepare_finish(int rows, int output_channels, int input_channels, Type output_type,
                        const ops::Epilogue &, QuantizationGroups groups = {});
    void linear(RowScaledView input, RowScaledView weight, TensorView output);
    bool linear_finish(RowScaledView input, RowScaledView weight, TensorView output,
                       const ops::Epilogue &);
    // Optional encoded output is the normalized input row, not the projection.
    // A false result leaves the caller's existing execution path responsible.
    bool normalized_linear(TensorView input, TensorView normalized,
                           TensorView norm_weight, TensorView norm_bias, float epsilon,
                           const ops::NormFusion &, RowScaledView weight, TensorView output,
                           const ops::Epilogue &, RowScaledView encoded = {},
                           const QkvProjectionOutput *writeback = nullptr);
};
} // namespace tts
