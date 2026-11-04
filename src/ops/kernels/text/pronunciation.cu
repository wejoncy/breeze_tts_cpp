#include "ops/cuda/backend.h"
#include "ops/pronunciation.h"
#include "runtime/device.h"
#include <cub/block/block_reduce.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <climits>

namespace tts::ops {
namespace {
struct Maximum {
    float value;
    int index;
};
struct FirstMaximum {
    __device__ Maximum operator()(Maximum a, Maximum b) const {
        return b.value > a.value || (b.value == a.value && b.index < a.index) ? b : a;
    }
};
using Reduce = cub::BlockReduce<Maximum, 256>;
struct View {
    void *data;
    int rows, cols, stride;
    explicit View(TensorView v) : data(v.data), rows(v.rows), cols(v.cols), stride(v.ld()) {}
};
template <class Bias>
__global__ void selection_kernel(View logits, View pos_logits, const Bias *bias,
                                  const Bias *pos_bias, const float *descriptor_bias,
                                  View char_descriptor, View second_order_descriptor,
                                  View legal, const int *positions, const int *char_ids,
                                  int *output_ids) {
    int row = blockIdx.x, ch = char_ids[row], tid = threadIdx.x;
    if (positions[row] < 0 || positions[row] >= logits.rows || ch < 0 || ch >= char_descriptor.rows) {
        if (!tid)
            output_ids[row] = -1;
        return;
    }
    __shared__ Reduce::TempStorage temporary;
    __shared__ int pos_id;
    __shared__ float row_max;
    auto *pos_row = static_cast<const float *>(pos_logits.data) + size_t(row) * pos_logits.stride;
    Maximum local{-CUDART_INF_F, INT_MAX};
    FirstMaximum maximum;
    for (int c = tid; c < pos_logits.cols; c += blockDim.x)
        local = maximum(local, {pos_row[c] + float(pos_bias[c]), c});
    Maximum result = Reduce(temporary).Reduce(local, maximum);
    if (!tid)
        pos_id = result.index;
    __syncthreads();
    auto *logit_row = static_cast<const float *>(logits.data) + size_t(row) * logits.stride;
    local = {-CUDART_INF_F, INT_MAX};
    for (int c = tid; c < logits.cols; c += blockDim.x)
        local = maximum(local, {logit_row[c] + float(bias[c]), c});
    result = Reduce(temporary).Reduce(local, maximum);
    if (!tid)
        row_max = result.value;
    __syncthreads();
    auto *first = static_cast<const float *>(char_descriptor.data) + size_t(ch) * char_descriptor.stride;
    auto *second = static_cast<const float *>(second_order_descriptor.data) +
                   (size_t(ch) * pos_logits.cols + pos_id) * second_order_descriptor.stride;
    auto *mask = static_cast<const uint32_t *>(legal.data) + size_t(ch) * legal.stride;
    local = {-CUDART_INF_F, INT_MAX};
    for (int c = tid; c < logits.cols; c += blockDim.x) {
        float score = 0.f;
        if (mask[c / 32] & (uint32_t(1) << (c % 32))) {
            float descriptor = (descriptor_bias[c] + first[c]) + second[c];
            float probability = 1.f / (1.f + expf(-descriptor));
            score = expf((logit_row[c] + float(bias[c])) - row_max) * probability;
        }
        local = maximum(local, {score, c});
    }
    // Normalization divides every label by the same positive scalar and does
    // not change its argmax. No probabilities or normalization scratch needed.
    result = Reduce(temporary).Reduce(local, maximum);
    if (!tid)
        output_ids[row] = result.index;
}
} // namespace
void select_pronunciation( TensorView logits, TensorView pos,
                          TensorView bias, TensorView pos_bias, TensorView descriptor_bias,
                          TensorView chars, TensorView second, TensorView legal,
                          const int *positions, const int *char_ids, int *output_ids) {
    const auto stream = execution_context(logits).stream;
    if (!logits.data || !pos.data || !bias.data || !pos_bias.data || !descriptor_bias.data ||
        !chars.data || !second.data || !legal.data || !positions || !char_ids || !output_ids ||
        logits.rows < 1 || logits.cols < 1 || pos.cols < 1 || pos.rows != logits.rows ||
        logits.type != Type::F32 || pos.type != Type::F32 ||
        (bias.type != Type::F16 && bias.type != Type::BF16) || pos_bias.type != bias.type ||
        bias.numel() != size_t(logits.cols) || pos_bias.numel() != size_t(pos.cols) ||
        descriptor_bias.type != Type::F32 || descriptor_bias.rows != 1 ||
        descriptor_bias.cols != logits.cols || chars.type != Type::F32 ||
        second.type != Type::F32 || chars.cols != logits.cols || second.cols != logits.cols ||
        int64_t(second.rows) != int64_t(chars.rows) * pos.cols || legal.type != Type::I32 ||
        legal.rows != chars.rows || legal.cols != (logits.cols - 1) / 32 + 1)
        throw std::invalid_argument("invalid pronunciation selection geometry/type");
    if (bias.type == Type::F16)
        selection_kernel<<<logits.rows, 256, 0, stream>>>(
            View(logits), View(pos), bias.ptr<const __half>(), pos_bias.ptr<const __half>(),
            descriptor_bias.ptr<const float>(), View(chars), View(second), View(legal), positions, char_ids, output_ids);
    else
        selection_kernel<<<logits.rows, 256, 0, stream>>>(
            View(logits), View(pos), bias.ptr<const __nv_bfloat16>(), pos_bias.ptr<const __nv_bfloat16>(),
            descriptor_bias.ptr<const float>(), View(chars), View(second), View(legal), positions, char_ids, output_ids);
    CUDA(cudaGetLastError());
}
} // namespace tts::ops
