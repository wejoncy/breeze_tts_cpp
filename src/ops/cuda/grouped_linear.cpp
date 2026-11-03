#include "ops/backend.h"
#include "ops/cuda/lt_linear.h"

namespace tts {
void Backend::grouped_linear(TensorView x, TensorView weight, TensorView output, int groups) {
    if ((output.type != Type::F32 && (groups != 1 || output.type != x.type)) ||
        !lt_linear_->linear(x, weight, output, x.type == Type::F32, 0, 1,
                            blas_workspace_.data(), blas_workspace_.size(), stream, groups))
        throw std::invalid_argument("invalid grouped linear views");
}
} // namespace tts
