#include "kernels/attention/flash_attention.h"
#include "runtime/device.h"
#define FLASH_NAMESPACE tts_flash
#include "flash-attention/flash.h"
#include "flash-attention/flash_fwd_kernel.h"

namespace tts {
template <int Dim, bool Causal, bool Local>
__global__ void flash_attention_prefill(const __grid_constant__ tts_flash::Flash_fwd_params p) {
    using Traits = Flash_fwd_kernel_traits<Dim, 64, 64, 4, false, false, cutlass::bfloat16_t>;
    tts_flash::compute_attn<Traits, false, Causal, Local, false, false, true, false, false>(p);
}
__global__ void
flash_attention_paged_prefill(const __grid_constant__ tts_flash::Flash_fwd_params p) {
    using Traits = Flash_fwd_kernel_traits<128, 64, 64, 4, false, false, cutlass::bfloat16_t>;
    tts_flash::compute_attn_splitkv<Traits, true, false, false, false, true, false, false, false>(
        p);
}
__global__ void
flash_attention_paged_decode(const __grid_constant__ tts_flash::Flash_fwd_params base,
                             const int *positions, CacheLayout cache) {
    auto p = base;
    // Positions advance on device inside the frame graph. Specialize the key
    // length per CTA without a metadata kernel or host synchronization.
    if (cache.sequences) {
        const auto &sequence = cache.sequences[blockIdx.y];
        if (!sequence.position) {
            auto *out = static_cast<__nv_bfloat16 *>(p.o_ptr) +
                        (size_t(blockIdx.y) * p.h + blockIdx.z) * 128;
            for (int d = threadIdx.x; d < 128; d += blockDim.x)
                out[d] = __float2bfloat16(0.f);
            return;
        }
        if (!cache.key_pages) {
            p.k_ptr = static_cast<char *>(sequence.keys) + cache.layer * sequence.layer_bytes;
            p.v_ptr = static_cast<char *>(sequence.values) + cache.layer * sequence.layer_bytes;
        }
        p.block_table = const_cast<int *>(sequence.blocks);
        p.block_table_batch_stride = 0;
        p.seqlen_k = *sequence.position + 1;
    } else
        p.seqlen_k = positions[blockIdx.y] + 1;
    using Traits = Flash_fwd_kernel_traits<128, 64, 64, 4, false, false, cutlass::bfloat16_t>;
    tts_flash::compute_attn_splitkv<Traits, true, false, false, false, true, false, false, false>(
        p);
}
template <int Dim, bool Causal, bool Local> static void configure() {
    using Traits = Flash_fwd_kernel_traits<Dim, 64, 64, 4, false, false, cutlass::bfloat16_t>;
    CUDA(cudaFuncSetAttribute(flash_attention_prefill<Dim, Causal, Local>,
                              cudaFuncAttributeMaxDynamicSharedMemorySize, Traits::kSmemSize));
}
void initialize_flash_attention() {
    configure<128, true, false>();
    configure<256, false, false>();
    configure<256, false, true>();
    using Traits = Flash_fwd_kernel_traits<128, 64, 64, 4, false, false, cutlass::bfloat16_t>;
    CUDA(cudaFuncSetAttribute(flash_attention_paged_decode,
                              cudaFuncAttributeMaxDynamicSharedMemorySize, Traits::kSmemSize));
    CUDA(cudaFuncSetAttribute(flash_attention_paged_prefill,
                              cudaFuncAttributeMaxDynamicSharedMemorySize, Traits::kSmemSize));
}
template <int Dim, bool Causal, bool Local>
static void launch(cudaStream_t stream, tts_flash::Flash_fwd_params &p) {
    using Traits = Flash_fwd_kernel_traits<Dim, 64, 64, 4, false, false, cutlass::bfloat16_t>;
    dim3 grid((p.seqlen_q + 63) / 64, p.b, p.h);
    flash_attention_prefill<Dim, Causal, Local>
        <<<grid, Traits::kNThreads, Traits::kSmemSize, stream>>>(p);
}
void flash_prefill(cudaStream_t stream, TensorView q, TensorView k, TensorView v, TensorView out,
                   float *lse, int batch, int time, int slot, int cap, int heads, int kvheads,
                   int dim, bool causal, int window, const PackedBatch *packed, CacheLayout cache,
                   int key_time) {
    if (key_time && (packed || key_time < time || key_time > cap))
        throw std::invalid_argument("invalid FlashAttention key length");
    tts_flash::Flash_fwd_params p{};
    p.q_ptr = q.data;
    p.o_ptr = out.data;
    size_t offset = size_t(slot) * cap * kvheads * dim * 2;
    p.k_ptr = k.data ? static_cast<char *>(k.data) + offset : nullptr;
    p.v_ptr = v.data ? static_cast<char *>(v.data) + offset : nullptr;
    p.q_batch_stride = p.o_batch_stride = int64_t(time) * heads * dim;
    p.k_batch_stride = p.v_batch_stride = int64_t(cap) * kvheads * dim;
    p.q_row_stride = p.o_row_stride = heads * dim;
    p.k_row_stride = p.v_row_stride = kvheads * dim;
    p.q_head_stride = p.k_head_stride = p.v_head_stride = p.o_head_stride = dim;
    p.b = batch;
    p.h = heads;
    p.h_k = kvheads;
    p.h_h_k_ratio = heads / kvheads;
    p.d = dim;
    p.d_rounded = dim;
    p.seqlen_q = time;
    p.seqlen_k = key_time ? key_time : time;
    p.seqlen_q_rounded = (time + 127) / 128 * 128;
    p.seqlen_k_rounded = (p.seqlen_k + 127) / 128 * 128;
    p.scale_softmax = 1.f / std::sqrt(float(dim));
    p.scale_softmax_log2 = p.scale_softmax * 1.4426950408889634f;
    p.p_dropout = p.rp_dropout = 1.f;
    p.p_dropout_in_uint8_t = 255;
    p.scale_softmax_rp_dropout = p.scale_softmax;
    p.is_bf16 = true;
    p.is_causal = causal;
    p.is_seqlens_k_cumulative = true;
    p.window_size_left = window > 0 ? window : -1;
    p.window_size_right = causal ? 0 : (window > 0 ? window : -1);
    p.softmax_lse_ptr = lse;
    if (packed) {
        p.cu_seqlens_q = const_cast<int *>(packed->offsets);
        p.seqused_k = const_cast<int *>(packed->lengths);
        p.total_q = packed->rows;
        p.unpadded_lse = true;
    }
    if (cache.block_table) {
        if (dim != 128 || !causal || !cache.valid_paged())
            throw std::invalid_argument("unsupported paged FlashAttention layout");
        p.k_ptr = k.data;
        p.v_ptr = v.data;
        p.k_batch_stride = p.v_batch_stride = int64_t(cache.page_size) * kvheads * dim;
        p.block_table = const_cast<int *>(cache.block_table) + slot * cache.block_stride;
        p.block_table_batch_stride = cache.block_stride;
        p.page_block_size = cache.page_size;
        p.k_pages = cache.key_pages;
        p.v_pages = cache.value_pages;
        p.page_layer_offset = cache.layer * cache.page_layer_bytes / sizeof(cutlass::bfloat16_t);
        using Traits = Flash_fwd_kernel_traits<128, 64, 64, 4, false, false, cutlass::bfloat16_t>;
        dim3 grid((time + 63) / 64, batch, heads);
        flash_attention_paged_prefill<<<grid, Traits::kNThreads, Traits::kSmemSize, stream>>>(p);
        return;
    }
    if (dim == 128 && causal)
        launch<128, true, false>(stream, p);
    else if (dim == 256 && !causal) {
        if (window > 0)
            launch<256, false, true>(stream, p);
        else
            launch<256, false, false>(stream, p);
    } else
        throw std::runtime_error("unsupported FlashAttention specialization");
}
void flash_decode(cudaStream_t stream, TensorView q, TensorView k, TensorView v, TensorView out,
                  float *lse, const int *positions, int batch, int heads, int kvheads,
                  CacheLayout cache) {
    tts_flash::Flash_fwd_params p{};
    p.q_ptr = q.data;
    p.k_ptr = k.data;
    p.v_ptr = v.data;
    p.o_ptr = out.data;
    p.q_batch_stride = p.o_batch_stride = p.q_row_stride = p.o_row_stride = heads * 128;
    p.k_batch_stride = p.v_batch_stride = int64_t(cache.page_size) * kvheads * 128;
    p.k_row_stride = p.v_row_stride = kvheads * 128;
    p.q_head_stride = p.k_head_stride = p.v_head_stride = p.o_head_stride = 128;
    p.b = batch;
    p.h = heads;
    p.h_k = kvheads;
    p.h_h_k_ratio = heads / kvheads;
    p.d = p.d_rounded = 128;
    p.seqlen_q = 1;
    p.seqlen_q_rounded = 128;
    p.scale_softmax = 1.f / std::sqrt(128.f);
    p.scale_softmax_log2 = p.scale_softmax * 1.4426950408889634f;
    p.p_dropout = p.rp_dropout = 1.f;
    p.p_dropout_in_uint8_t = 255;
    p.scale_softmax_rp_dropout = p.scale_softmax;
    p.is_bf16 = p.is_causal = p.is_seqlens_k_cumulative = true;
    p.window_size_left = -1;
    p.window_size_right = 0;
    p.softmax_lse_ptr = lse;
    p.block_table = const_cast<int *>(cache.block_table);
    p.block_table_batch_stride = cache.block_stride;
    p.page_block_size = cache.page_size;
    p.k_pages = cache.key_pages;
    p.v_pages = cache.value_pages;
    p.page_layer_offset = cache.layer * cache.page_layer_bytes / sizeof(cutlass::bfloat16_t);
    using Traits = Flash_fwd_kernel_traits<128, 64, 64, 4, false, false, cutlass::bfloat16_t>;
    flash_attention_paged_decode<<<dim3(1, batch, heads), Traits::kNThreads, Traits::kSmemSize,
                                   stream>>>(p, positions, cache);
}
} // namespace tts
