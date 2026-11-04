#include "runtime/device.h"
#define FLASH_NAMESPACE tts_flash
#include "flash-attention/flash.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

namespace tts {
// FA2's FP32 log-sum-exp merge specialized to decode's single query. A warp
// owns one output head; no split-count or head-dimension template family.
template <class Element>
__global__ void combine_decode(const __grid_constant__ tts_flash::Flash_fwd_params p) {
    int lane = threadIdx.x % 32;
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= p.b * p.h) return;
    const auto *lse = static_cast<const float *>(p.softmax_lseaccum_ptr);
    float local[4], maximum = -CUDART_INF_F;
    #pragma unroll
    for (int group = 0; group < 4; ++group) {
        int split = group * 32 + lane;
        local[group] = split < p.num_splits ? lse[size_t(split) * p.b * p.h + row] : -CUDART_INF_F;
        maximum = fmaxf(maximum, local[group]);
    }
    #pragma unroll
    for (int delta = 16; delta; delta /= 2)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, delta));
    if (maximum == -CUDART_INF_F) maximum = 0;
    float sum = 0;
    #pragma unroll
    for (int group = 0; group < 4; ++group)
        sum += expf(local[group] - maximum);
    #pragma unroll
    for (int delta = 16; delta; delta /= 2)
        sum += __shfl_xor_sync(0xffffffffu, sum, delta);
    float logsum = sum == 0 ? CUDART_INF_F : logf(sum) + maximum;
    if (!lane) static_cast<float *>(p.softmax_lse_ptr)[row] = logsum;
    float output[4] = {};
    const auto *partials = static_cast<const float *>(p.oaccum_ptr);
    #pragma unroll
    for (int group = 0; group < 4; ++group) {
        float own_scale = expf(local[group] - logsum);
        for (int i = 0; i < 32 && group * 32 + i < p.num_splits; ++i) {
            float scale = __shfl_sync(0xffffffffu, own_scale, i);
            size_t source = (size_t(group * 32 + i) * p.b * p.h + row) * p.d_rounded;
            #pragma unroll
            for (int column = 0; column < 4; ++column)
                if (lane + column * 32 < p.d)
                    output[column] += scale * partials[source + lane + column * 32];
        }
    }
    auto *destination = static_cast<Element *>(p.o_ptr) +
                        (row / p.h) * p.o_batch_stride + (row % p.h) * p.o_head_stride;
    #pragma unroll
    for (int column = 0; column < 4; ++column)
        if (lane + column * 32 < p.d) destination[lane + column * 32] = Element(output[column]);
}
void initialize_flash_decode_combine() {
    cudaFuncAttributes attributes;
    CUDA(cudaFuncGetAttributes(&attributes, combine_decode<__half>));
    CUDA(cudaFuncGetAttributes(&attributes, combine_decode<__nv_bfloat16>));
}
void combine_flash_decode(cudaStream_t stream, const tts_flash::Flash_fwd_params &p) {
    int blocks = (p.b * p.h + 3) / 4;
    if (p.is_bf16) combine_decode<__nv_bfloat16><<<blocks, 128, 0, stream>>>(p);
    else combine_decode<__half><<<blocks, 128, 0, stream>>>(p);
}
} // namespace tts
