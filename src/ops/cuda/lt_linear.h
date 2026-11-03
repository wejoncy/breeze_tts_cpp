#pragma once
#include "ops/cuda/lt_plan.h"
#include "core/tensor.h"
#include "runtime/prepared_cache.h"
#include <tuple>
#include <vector>

namespace tts {
// Host plans borrow the backend's handle/workspace. Initialization selects and
// warms each matrix family; serving changes only its exact row count.
class LtLinear {
    using Key = std::tuple<int, int, int, int, int, Type, Type, cublasComputeType_t,
                           cublasLtEpilogue_t, uintptr_t, std::array<uint32_t, 4>, float, float>;
    struct Plan {
        cublasLtMatmulDescOpaque_t operation{};
        cublasLtMatrixLayoutOpaque_t weight{}, input{}, output{};
        struct Sample { int rows; size_t algorithm, tail_algorithm; };
        struct Algorithm { cublasLtMatmulHeuristicResult_t configuration; int rows;  };
        std::vector<Algorithm> algorithms;
        std::vector<Sample> samples;
        std::vector<int> classic_rows; // Initialization execution metadata, sealed with the plan.
        cublasComputeType_t compute = CUBLAS_COMPUTE_32F;
        TensorView x, w, bias, y;
        float alpha = 1, beta = 0;
        int max_rows = 0;
        void rows(int value);
        const cublasLtMatmulAlgo_t *select(cublasLtHandle_t, int rows, size_t workspace_bytes);
    };
    cublasLtHandle_t handle_;
    cublasHandle_t blas_;
    PreparedCache<Key, Plan> plans_;
    bool matmul(TensorView x, TensorView w, TensorView bias, TensorView y,
                 cublasLtEpilogue_t epilogue, bool full_precision, float alpha, float beta,
                 void *workspace, size_t workspace_bytes, cudaStream_t stream, int groups, bool execute, int max_rows = 0);

  public:
    LtLinear(cublasLtHandle_t handle, cublasHandle_t blas) : handle_(handle), blas_(blas) {}
    void seal(void *workspace, size_t workspace_bytes, cudaStream_t stream);
    bool linear(TensorView x, TensorView w, TensorView y, bool full_precision,
                float accumulation, float product_scale, void *workspace,
                size_t workspace_bytes, cudaStream_t stream, int groups = 1);
    bool prepare(TensorView x, TensorView w, TensorView y, bool full_precision,
                 float accumulation, float product_scale, void *workspace,
                 size_t workspace_bytes, cudaStream_t stream, int groups = 1, int max_rows = 0,
                 TensorView bias = {}, cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT);
    bool gelu(TensorView x, TensorView w, TensorView bias, TensorView y, void *workspace,
              size_t workspace_bytes, cudaStream_t stream);
};
} // namespace tts
