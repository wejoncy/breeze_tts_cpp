#include "ops/cuda/lt_linear.h"
#include "ops/backend.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace tts {
namespace {
cudaDataType dtype(Type type) {
    return type == Type::F16 ? CUDA_R_16F : type == Type::BF16 ? CUDA_R_16BF : CUDA_R_32F;
}
uint32_t alignment(TensorView view, size_t batch_stride = 0) {
    const auto bits = reinterpret_cast<uintptr_t>(view.data) | size_t(view.ld()) * bytes(view.type) | batch_stride * bytes(view.type);
    return uint32_t(std::min<uintptr_t>(256, bits & (~bits + 1)));
}
std::string family(TensorView x, TensorView w, TensorView y) {
    return " N=" + std::to_string(w.rows) + " K=" + std::to_string(x.cols) +
           " xld=" + std::to_string(x.ld()) + " yld=" + std::to_string(y.ld()) +
           " types=" + std::to_string(int(x.type)) + "/" + std::to_string(int(y.type));
}
// Views have already passed matmul validation. The ordinary vendor path is
// permitted only for the exact launch metadata initialized before sealing.
void classic_matmul(cublasHandle_t handle, TensorView x, TensorView w, TensorView y,
                     cublasComputeType_t compute, float alpha, float beta) {
    const int groups = x.cols / w.cols, columns = w.rows / groups;
    if (groups == 1)
        BLAS(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, columns, x.rows, w.cols,
                          &alpha, w.data, dtype(w.type), w.cols, x.data, dtype(x.type), x.ld(),
                          &beta, y.data, dtype(y.type), y.ld(), compute,
                          CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    else
        BLAS(cublasGemmStridedBatchedEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N, columns, x.rows, w.cols,
            &alpha, w.data, dtype(w.type), w.cols, int64_t(columns) * w.cols,
            x.data, dtype(x.type), x.ld(), w.cols, &beta, y.data, dtype(y.type),
            y.ld(), columns, groups, compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
} // namespace
bool LtLinear::matmul(TensorView x, TensorView w, TensorView bias, TensorView y,
                       cublasLtEpilogue_t epilogue, bool full_precision, float alpha, float beta,
                       void *workspace, size_t workspace_bytes, cudaStream_t stream, int groups, bool execute, int max_rows) {
    if ((x.type != Type::F16 && x.type != Type::BF16 && x.type != Type::F32) ||
        x.type != w.type || (x.type != y.type && y.type != Type::F32) ||
        !std::isfinite(alpha) || !std::isfinite(beta) || !x.data || !w.data || !y.data || x.rows < 1 || x.cols < 1 || w.rows < 1 ||
        groups < 1 || x.cols % groups || w.rows % groups || x.cols / groups != w.cols || x.rows != y.rows || y.cols != w.rows || !w.dense())
        return false;
    if (epilogue == CUBLASLT_EPILOGUE_GELU_BIAS &&
        (groups != 1 || x.type == Type::F32 || y.type != x.type || bias.type != x.type || !bias.data ||
         bias.numel() != size_t(w.rows) || !bias.dense()))
        return false;
    const auto compute = x.type == Type::F32
                             ? (full_precision ? CUBLAS_COMPUTE_32F_PEDANTIC
                                               : CUBLAS_COMPUTE_32F_FAST_TF32)
                             : CUBLAS_COMPUTE_32F;
    const std::array<uint32_t, 4> alignments{
        alignment(w), alignment(x, groups > 1 ? w.cols : 0), alignment(y, groups > 1 ? w.rows / groups : 0), alignment(y, groups > 1 ? w.rows / groups : 0)};
    // Bias has no min-alignment preference, so keep its actual alignment class.
    const auto address = reinterpret_cast<uintptr_t>(bias.data);
    Key key{w.rows / groups, w.cols, groups, x.ld(), y.ld(), x.type, y.type, compute, epilogue,
            address & (~address + 1), alignments, alpha, beta};
    auto *plan = plans_.find(key);
    if (!plan) {
        if (plans_.sealed())
            throw std::logic_error("GEMM matrix family was not prepared during initialization" +
                                     family(x, w, y));
        plan = &plans_.prepare(key, [&] {
            auto p = std::make_unique<Plan>();
            p->compute = compute;
            BLAS(cublasLtMatmulDescInit(&p->operation, compute, CUDA_R_32F));
            auto transpose = CUBLAS_OP_T;
            BLAS(cublasLtMatmulDescSetAttribute(&p->operation, CUBLASLT_MATMUL_DESC_TRANSA,
                                                &transpose, sizeof(transpose)));
            BLAS(cublasLtMatmulDescSetAttribute(&p->operation, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                &epilogue, sizeof(epilogue)));
            if (bias.data) {
                auto bias_type = dtype(bias.type);
                BLAS(cublasLtMatmulDescSetAttribute(&p->operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                    &bias.data, sizeof(bias.data)));
                BLAS(cublasLtMatmulDescSetAttribute(&p->operation, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                                    &bias_type, sizeof(bias_type)));
            }
            BLAS(cublasLtMatrixLayoutInit(&p->weight, dtype(w.type), w.cols, w.rows / groups, w.cols));
            BLAS(cublasLtMatrixLayoutInit(&p->input, dtype(x.type), w.cols, x.rows, x.ld()));
            BLAS(cublasLtMatrixLayoutInit(&p->output, dtype(y.type), w.rows / groups, y.rows, y.ld()));
            if (groups > 1) {
                const std::array<cublasLtMatrixLayout_t, 3> layouts{&p->weight, &p->input, &p->output};
                const std::array<int64_t, 3> strides{int64_t(w.cols) * (w.rows / groups), w.cols, w.rows / groups};
                for (size_t i = 0; i < layouts.size(); ++i) {
                    BLAS(cublasLtMatrixLayoutSetAttribute(layouts[i], CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                          &groups, sizeof(groups)));
                    BLAS(cublasLtMatrixLayoutSetAttribute(layouts[i], CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                          &strides[i], sizeof(strides[i])));
                }
            }
            return p;
        });
    }
    if (plans_.sealed() && x.rows > plan->max_rows)
        throw std::out_of_range("GEMM row count exceeds initialized capacity" + family(x, w, y));
    const int capacity = max_rows ? max_rows : x.rows;
    if (capacity < x.rows) throw std::invalid_argument("GEMM planning capacity below sample rows");
    if (!plans_.sealed()) {
        // Query each initialized M once, whether it comes from a capacity
        // sample or a naturally encountered execution/graph shape.
        auto prepare = [&](int rows) {
            auto sample = std::find_if(plan->samples.begin(), plan->samples.end(),
                                       [&](const auto &entry) { return entry.rows == rows; });
            if (sample != plan->samples.end()) return sample->algorithm;
            plan->rows(rows);
            cublasLtMatmulHeuristicResult_t selected{};
            if (!select_lt_matmul(handle_, &plan->operation, &plan->weight, &plan->input,
                                   &plan->output, selected, workspace_bytes, alignments))
                throw std::runtime_error("no initialized Lt GEMM algorithm for matrix family" + family(x, w, y));
            size_t index = 0;
            for (; index < plan->algorithms.size(); ++index)
                if (!std::memcmp(&selected.algo, &plan->algorithms[index].configuration.algo, sizeof(selected.algo)))
                    break;
            if (index == plan->algorithms.size())
                plan->algorithms.push_back({selected, rows});
            plan->samples.push_back({rows, index, index});
            return index;
        };
        if (capacity > plan->max_rows) {
            // Retain earlier samples/configs when initialization expands the
            // capacity; only new points need another vendor query.
            for (int rows = 1;; rows = int(std::min<int64_t>(capacity, int64_t(rows) * 2))) {
                auto index = prepare(rows);
                auto tail = rows < capacity ? prepare(rows + 1) : index;
                auto sample = std::find_if(plan->samples.begin(), plan->samples.end(),
                                           [&](const auto &entry) { return entry.rows == rows; });
                sample->tail_algorithm = tail;
                if (rows == capacity) break;
            }
            plan->max_rows = capacity;
            plan->x = x; plan->w = w; plan->bias = bias; plan->y = y;
            plan->alpha = alpha; plan->beta = beta;
        }
        prepare(x.rows);
    }
    if (execute && epilogue == CUBLASLT_EPILOGUE_DEFAULT &&
        (!plans_.sealed() || std::binary_search(plan->classic_rows.begin(), plan->classic_rows.end(), x.rows))) {
        classic_matmul(blas_, x, w, y, compute, alpha, beta);
        if (!plans_.sealed()) {
            auto position = std::lower_bound(plan->classic_rows.begin(), plan->classic_rows.end(), x.rows);
            if (position == plan->classic_rows.end() || *position != x.rows)
                plan->classic_rows.insert(position, x.rows);
        }
        return true;
    }
    plan->rows(x.rows);
    if (bias.data)
        BLAS(cublasLtMatmulDescSetAttribute(&plan->operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                            &bias.data, sizeof(bias.data)));
    if (!execute) return true;
    auto *algorithm = plan->select(handle_, x.rows, workspace_bytes);
    BLAS(cublasLtMatmul(handle_, &plan->operation, &alpha, w.data, &plan->weight, x.data,
                        &plan->input, &beta, y.data, &plan->output, y.data, &plan->output,
                        algorithm, workspace, workspace_bytes, stream));
    return true;
}
void LtLinear::Plan::rows(int value) {
    const uint64_t count = value;
    BLAS(cublasLtMatrixLayoutSetAttribute(&input, CUBLASLT_MATRIX_LAYOUT_COLS, &count, sizeof(count)));
    BLAS(cublasLtMatrixLayoutSetAttribute(&output, CUBLASLT_MATRIX_LAYOUT_COLS, &count, sizeof(count)));
}
const cublasLtMatmulAlgo_t *LtLinear::Plan::select(cublasLtHandle_t handle, int value, size_t workspace_bytes) {
    size_t best = 0;
    double distance = std::numeric_limits<double>::infinity();
    for (const auto &sample : samples) {
        const double score = double(std::max(sample.rows, value)) / std::min(sample.rows, value);
        if (score < distance) {
            distance = score;
            best = value > sample.rows ? sample.tail_algorithm : sample.algorithm;
        }
    }
    auto compatible = [&](size_t index) {
        cublasLtMatmulHeuristicResult_t checked{};
        if (cublasLtMatmulAlgoCheck(handle, &operation, &weight, &input, &output, &output,
                                    &algorithms[index].configuration.algo, &checked) != CUBLAS_STATUS_SUCCESS ||
            checked.state != CUBLAS_STATUS_SUCCESS || checked.workspaceSize > workspace_bytes)
            return false;
        return true;
    };
    if (compatible(best)) return &algorithms[best].configuration.algo;
    for (size_t index = 0; index < algorithms.size(); ++index)
        if (index != best && compatible(index)) return &algorithms[index].configuration.algo;
    throw std::runtime_error("no prepared Lt GEMM algorithm supports M=" + std::to_string(value));
}
void LtLinear::seal(void *workspace, size_t workspace_bytes, cudaStream_t stream) {
    if (plans_.sealed()) return;
    CUDA(cudaStreamSynchronize(stream));
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    bool capturing = false;
    try {
        CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        capturing = true;
        plans_.prepare_all([&](Plan &plan) {
            if (plan.bias.data)
                BLAS(cublasLtMatmulDescSetAttribute(&plan.operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                    &plan.bias.data, sizeof(plan.bias.data)));
            for (int rows : plan.classic_rows) {
                auto x = plan.x, y = plan.y;
                x.rows = y.rows = rows;
                classic_matmul(blas_, x, plan.w, y, plan.compute, plan.alpha, plan.beta);
            }
            // The same opaque config can have launch branches at different M;
            // upload every initialized sample, retaining config deduplication.
            for (const auto &sample : plan.samples) {
                const auto &algorithm = plan.algorithms[sample.algorithm];
                plan.rows(sample.rows);
                BLAS(cublasLtMatmul(handle_, &plan.operation, &plan.alpha, plan.w.data, &plan.weight,
                                    plan.x.data, &plan.input, &plan.beta, plan.y.data, &plan.output,
                                    plan.y.data, &plan.output, &algorithm.configuration.algo,
                                    workspace, workspace_bytes, stream));
            }
            plan.rows(plan.max_rows);
        });
        const auto capture_status = cudaStreamEndCapture(stream, &graph);
        capturing = false;
        CUDA(capture_status);
        CUDA(cudaGraphInstantiate(&executable, graph, 0));
        CUDA(cudaGraphUpload(executable, stream));
        CUDA(cudaStreamSynchronize(stream));
        // Upload prepares functions and launch resources without executing any
        // warm GEMM or writing buffers borrowed from the initialization arena.
        CUDA(cudaGraphExecDestroy(executable)); executable = nullptr;
        CUDA(cudaGraphDestroy(graph)); graph = nullptr;
    } catch (...) {
        if (capturing) cudaStreamEndCapture(stream, &graph);
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
        throw;
    }
    plans_.seal();
}
bool LtLinear::linear(TensorView x, TensorView w, TensorView y, bool full_precision,
                       float accumulation, float product_scale, void *workspace,
                       size_t workspace_bytes, cudaStream_t stream, int groups) {
    return matmul(x, w, {}, y, CUBLASLT_EPILOGUE_DEFAULT, full_precision, product_scale,
                    accumulation, workspace, workspace_bytes, stream, groups, true);
}
bool LtLinear::prepare(TensorView x, TensorView w, TensorView y, bool full_precision,
                        float accumulation, float product_scale, void *workspace,
                        size_t workspace_bytes, cudaStream_t stream, int groups, int max_rows,
                        TensorView bias, cublasLtEpilogue_t epilogue) {
    if (plans_.sealed()) throw std::logic_error("linear preparation after sealing");
    return matmul(x, w, bias, y, epilogue, full_precision, product_scale,
                  accumulation, workspace, workspace_bytes, stream, groups, false, max_rows);
}
void Backend::prepare_linear(TensorView x, TensorView w, TensorView y, bool full_precision,
                             float accumulation, float product_scale, int groups, int max_rows) {
    if (!lt_linear_->prepare(x, w, y, full_precision, accumulation, product_scale,
                             blas_workspace_.data(), blas_workspace_.size(), stream, groups, max_rows))
        throw std::invalid_argument("invalid linear planning views");
}
bool Backend::prepare_linear_gelu(TensorView x, TensorView w, TensorView bias, TensorView y,
                                  int max_rows) {
    return lt_linear_->prepare(x, w, y, false, 0, 1, blas_workspace_.data(),
                                blas_workspace_.size(), stream, 1, max_rows, bias,
                                CUBLASLT_EPILOGUE_GELU_BIAS);
}
bool LtLinear::gelu(TensorView x, TensorView w, TensorView bias, TensorView y, void *workspace,
                    size_t workspace_bytes, cudaStream_t stream) {
    return matmul(x, w, bias, y, CUBLASLT_EPILOGUE_GELU_BIAS, false, 1, 0,
                    workspace, workspace_bytes, stream, 1, true);
}
bool Backend::linear_gelu(TensorView x, TensorView w, TensorView bias, TensorView y) {
    return lt_linear_->gelu(x, w, bias, y, blas_workspace_.data(), blas_workspace_.size(), stream);
}
void Backend::seal_plans() {
    lt_linear_->seal(blas_workspace_.data(), blas_workspace_.size(), stream);
    if (quant_)
        quant_->seal();
    plans_sealed_ = true;
}
} // namespace tts
