#include "runtime/cuda_context.h"
#include "ops/cuda/lt_linear.h"

namespace tts::ops {
void grouped_matmul(TensorView x, TensorView weight, TensorView output, int groups) {
    auto &gpu = execution_context(x, {weight, output});
    if ((output.type != Type::F32 && (groups != 1 || output.type != x.type)) ||
        !gpu.matmul->linear(x, weight, output, x.type == Type::F32, 0, 1,
                            gpu.gemm_workspace.data(), gpu.gemm_workspace.size(), gpu.stream, groups))
        throw std::invalid_argument("invalid grouped linear views");
}
} // namespace tts
