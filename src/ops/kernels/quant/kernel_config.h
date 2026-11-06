#pragma once
#include "core/kernel_problem.h"
#include <cstddef>
namespace tts {
enum class LinearKind { Lt, Gemv, Sm80, Hopper, Grouped };
struct LinearPath {
    LinearKind kind = LinearKind::Lt;
    int tokens = 1, channels = 0, splits = 1;
};
// Dynamic operand/scale storage plus the conservative split reduction storage.
inline size_t gemv_shared_bytes(KernelProblem problem, int tokens, int columns) {
    size_t input_groups = problem.input_group ?
        (problem.reduction + problem.input_group - 1) / problem.input_group : 0;
    size_t weight_groups = problem.weight_group ?
        (problem.reduction + problem.weight_group - 1) / problem.weight_group : 0;
    return size_t(tokens) * problem.reduction +
        sizeof(float) * (tokens * input_groups + columns * weight_groups + 8 * tokens);
}
} // namespace tts
