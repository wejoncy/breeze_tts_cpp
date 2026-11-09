#pragma once
#include "core/quantization.h"

namespace tts::ops {
// Explicit CPU conversion for checkpoint coefficients and reference preparation.
void convert_host(TensorView source, TensorView destination);
float rounded(float value, Type type);

void synchronize(int device);
// Shared by checkpoint loading and dynamic activation producers. Dispatch uses
// tensor dtype and quantization layout, independently of GEMM planning/model state.
void quantize_rows(TensorView input, RowScaledView output);
} // namespace tts::ops
