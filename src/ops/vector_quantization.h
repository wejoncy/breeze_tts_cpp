#pragma once
#include "core/tensor.h"

namespace tts {
// Nearest-code selection and optional residual update share one pass. Expanded
// distances retain low-precision arithmetic boundaries when discrete IDs depend
// on rounding; reduced distances omit the code-independent input norm.
void vector_quantize( TensorView input, TensorView scores, TensorView table,
                     TensorView norms, int *codes, int codebook, int codebooks,
                     bool subtract = true, bool expanded_distances = false);
} // namespace tts
