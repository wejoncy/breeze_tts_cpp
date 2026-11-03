#pragma once
#include "ops/cuda/blas_check.h"
#include "core/kernel_problem.h"
#include <cublasLt.h>
#include <array>
#include <cstdint>

namespace tts {
// One initialization-only selection path for both owned descriptors and opaque
// descriptors embedded in an already reserved host plan.
inline bool select_lt_matmul(cublasLtHandle_t handle, cublasLtMatmulDesc_t operation,
                             cublasLtMatrixLayout_t weight, cublasLtMatrixLayout_t input,
                             cublasLtMatrixLayout_t output,
                             cublasLtMatmulHeuristicResult_t &algorithm, size_t workspace_bytes,
                             const std::array<uint32_t, 4> &alignments) {
    cublasLtMatmulPreference_t preference = nullptr;
    BLAS(cublasLtMatmulPreferenceCreate(&preference));
    int count = 0;
    auto status = cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes,
        sizeof(workspace_bytes));
    size_t i = 0;
    for (auto attribute : {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                           CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                           CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
                           CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES}) {
        if (status == CUBLAS_STATUS_SUCCESS)
            status = cublasLtMatmulPreferenceSetAttribute(preference, attribute, &alignments[i],
                                                          sizeof(alignments[i]));
        ++i;
    }
    if (status == CUBLAS_STATUS_SUCCESS)
        status = cublasLtMatmulAlgoGetHeuristic(handle, operation, weight, input, output,
                                                output, preference, 1, &algorithm, &count);
    cublasLtMatmulPreferenceDestroy(preference);
    if (status == CUBLAS_STATUS_NOT_SUPPORTED)
        return false;
    BLAS(status);
    return count && algorithm.state == CUBLAS_STATUS_SUCCESS &&
           algorithm.workspaceSize <= workspace_bytes;
}
struct LtPlan {
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t weight = nullptr, input = nullptr, output = nullptr;
    cublasLtMatmulHeuristicResult_t algorithm{};
    LibraryWork work;
    LtPlan() = default;
    LtPlan(const LtPlan &) = delete;
    ~LtPlan() {
        if (operation)
            cublasLtMatmulDescDestroy(operation);
        if (weight)
            cublasLtMatrixLayoutDestroy(weight);
        if (input)
            cublasLtMatrixLayoutDestroy(input);
        if (output)
            cublasLtMatrixLayoutDestroy(output);
    }
    bool select(cublasLtHandle_t handle, size_t workspace_bytes, uint32_t alignment = 256) {
        work = {};
        if (!select_lt_matmul(handle, operation, weight, input, output, algorithm,
                               workspace_bytes, {alignment, alignment, alignment, alignment}))
            return false;
        work.effective_waves = algorithm.wavesCount;
        int32_t splits = -1;
        uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_MASK;
        size_t written = 0;
        auto split_status = cublasLtMatmulAlgoConfigGetAttribute(
            &algorithm.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
            &splits, sizeof(splits), &written);
        auto reduction_status = cublasLtMatmulAlgoConfigGetAttribute(
            &algorithm.algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
            &reduction, sizeof(reduction), &written);
        // A serial dot product without reduction workspace has no parallel
        // reduction phase. In-place counters, parallel reduction, Stream-K
        // workspace and unknown metadata retain the conservative estimate.
        if (split_status == CUBLAS_STATUS_SUCCESS &&
            reduction_status == CUBLAS_STATUS_SUCCESS && splits >= 0 && splits <= 1 &&
            reduction == CUBLASLT_REDUCTION_SCHEME_NONE && !algorithm.workspaceSize)
            work.kernel_launches = 1;
        return true;
    }
};
} // namespace tts
