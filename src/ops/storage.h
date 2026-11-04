#pragma once
#include "core/quantization.h"

namespace tts::ops {
// Explicit CPU conversion for checkpoint coefficients and reference preparation.
void convert_host(TensorView source, TensorView destination);
float rounded(float value, Type type);

void upload_bytes(int device, void *destination, const void *source, size_t bytes);
void synchronize(int device);
void download_bytes(TensorView source, void *destination, size_t bytes);
void quantize_rows(TensorView input, RowScaledView output);
} // namespace tts::ops
