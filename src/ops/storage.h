#pragma once
#include "core/quantization.h"

namespace tts::ops {
void upload_bytes(int device, void *destination, const void *source, size_t bytes);
void synchronize(int device);
void download_bytes(TensorView source, void *destination, size_t bytes);
void quantize_rows(TensorView input, RowScaledView output);
} // namespace tts::ops
