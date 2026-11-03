#include "ops/attention.h"
#include "runtime/device.h"
#include <type_traits>
#define FLASH_NAMESPACE tts_flash
#include "flash-attention/flash.h"
#include "flash-attention/flash_fwd_kernel.h"

namespace tts {
void initialize_flash_decode_combine();
void combine_flash_decode(cudaStream_t, const tts_flash::Flash_fwd_params &);
// Each CTA resolves its sequence metadata before entering FlashAttention. The
// library still owns tiled MMA, online softmax and paged loads; no gather or
// sequence-length update kernel is needed for a graph replay.
__device__ bool sequence_params(tts_flash::Flash_fwd_params &p,
                                 DenseAttentionLayout dense, CacheLayout cache,
                                 const int *positions, bool decode, int lane) {
    if (dense.sequences) {
        const auto &s = dense.sequences[lane];
        if (s.query_length <= 0 || s.key_length <= 0 || blockIdx.x * 64 >= s.query_length)
            return false;
        p.q_ptr = static_cast<char *>(p.q_ptr) + size_t(s.query_offset) * p.q_row_stride * 2;
        p.o_ptr = static_cast<char *>(p.o_ptr) + size_t(s.query_offset) * p.o_row_stride * 2;
        p.q_batch_stride = p.o_batch_stride = 0;
        if (!cache.page_size) {
            p.k_ptr = static_cast<char *>(p.k_ptr) + size_t(s.key_offset) * p.k_row_stride * 2;
            p.v_ptr = static_cast<char *>(p.v_ptr) + size_t(s.key_offset) * p.v_row_stride * 2;
            p.k_batch_stride = p.v_batch_stride = 0;
            if (dense.key_sources) p.k_ptr = const_cast<void *>(dense.key_sources[lane]);
            if (dense.value_sources) p.v_ptr = const_cast<void *>(dense.value_sources[lane]);
        }
        // Each sequence owns [heads, query_length] at its packed row offset.
        // The rowblock entry receives local lane zero, so LSE needs only the
        // packed token capacity rather than sequences * maximum_length.
        p.softmax_lse_ptr = static_cast<float *>(p.softmax_lse_ptr) +
            size_t(s.query_offset) * p.h;
        if (cache.block_table) {
            p.block_table += size_t(lane) * p.block_table_batch_stride;
            p.block_table_batch_stride = 0;
        }
        p.b = 1;
        p.seqlen_q = s.query_length;
        p.seqlen_k = s.key_length;
    }
    if (cache.sequences) {
        const auto &s = cache.sequences[lane];
        if (!s.position) {
            // Decode's empty-K path writes 0/-inf partials on every replay.
            // No stale graph-padding data can survive into its FP32 merge.
            if (decode) p.seqlen_k = 0;
            return decode;
        }
        if (!cache.key_pages) {
            p.k_ptr = static_cast<char *>(s.keys) + cache.layer * s.layer_bytes;
            p.v_ptr = static_cast<char *>(s.values) + cache.layer * s.layer_bytes;
        }
        p.block_table = const_cast<int *>(s.blocks);
        p.block_table_batch_stride = 0;
        if (decode) p.seqlen_k = *s.position + 1;
    } else if (decode) {
        p.seqlen_k = positions[lane] + 1;
    }
    return true;
}
template <int Dim, bool Causal, bool Local, class Element = cutlass::bfloat16_t>
__global__ void flash_attention_prefill(const __grid_constant__ tts_flash::Flash_fwd_params base,
                                       DenseAttentionLayout dense) {
    auto p = base;
    if (!sequence_params(p, dense, {}, nullptr, false, blockIdx.y)) return;
    using Traits = Flash_fwd_kernel_traits<Dim, 64, 64, 4, false, false, Element>;
    tts_flash::compute_attn_1rowblock<Traits, false, Causal, Local, false, false, true, false, false>(
        p, dense.sequences ? 0 : blockIdx.y, blockIdx.z, blockIdx.x);
}
template <int Dim, bool Causal, class Element>
__global__ void flash_attention_paged_prefill(
    const __grid_constant__ tts_flash::Flash_fwd_params base,
    DenseAttentionLayout dense, CacheLayout cache) {
    auto p = base;
    if (!sequence_params(p, dense, cache, nullptr, false, blockIdx.y)) return;
    using Traits = Flash_fwd_kernel_traits<Dim, 64, 64, 4, false, false, Element>;
    tts_flash::compute_attn_1rowblock_splitkv<Traits, Causal, false, false, false, true, false, false, false>(
        p, dense.sequences ? 0 : blockIdx.y, blockIdx.z, blockIdx.x, 0, 1);
}
template <int Dim, class Element, bool Split = false>
__global__ void flash_attention_paged_decode(
    const __grid_constant__ tts_flash::Flash_fwd_params base,
    const int *positions, CacheLayout cache) {
    auto p = base;
    int lane = Split ? blockIdx.z / p.h : blockIdx.y;
    int head = Split ? blockIdx.z % p.h : blockIdx.z;
    if (!sequence_params(p, {}, cache, positions, true, lane)) return;
    using Traits = Flash_fwd_kernel_traits<Dim, 64, 64, 4, false, false, Element>;
    int splits = Split ? min(int(gridDim.y), max(1, (p.seqlen_k + 63) / 64)) : 1;
    tts_flash::compute_attn_1rowblock_splitkv<Traits, true, false, false, false, true,
                                             false, Split, false>(
        p, lane, head, 0, Split ? blockIdx.y : 0, splits);
}
template <int Dim, bool Causal, bool Local, class Element = cutlass::bfloat16_t>
static void configure() {
    using Traits = Flash_fwd_kernel_traits<Dim, 64, 64, 4, false, false, Element>;
    CUDA(cudaFuncSetAttribute(flash_attention_prefill<Dim, Causal, Local, Element>,
                              cudaFuncAttributeMaxDynamicSharedMemorySize, Traits::kSmemSize));
}
template <int Dim, class Element>
static void configure_dimension() {
    configure<Dim, false, false, Element>();
    configure<Dim, true, false, Element>();
    using Traits = Flash_fwd_kernel_traits<Dim, 64, 64, 4, false, false, Element>;
    CUDA(cudaFuncSetAttribute(flash_attention_paged_prefill<Dim, true, Element>,
                              cudaFuncAttributeMaxDynamicSharedMemorySize, Traits::kSmemSize));
    CUDA(cudaFuncSetAttribute(flash_attention_paged_prefill<Dim, false, Element>,
                              cudaFuncAttributeMaxDynamicSharedMemorySize, Traits::kSmemSize));
    CUDA(cudaFuncSetAttribute(flash_attention_paged_decode<Dim, Element>,
                              cudaFuncAttributeMaxDynamicSharedMemorySize, Traits::kSmemSize));
    CUDA(cudaFuncSetAttribute(flash_attention_paged_decode<Dim, Element, true>,
                              cudaFuncAttributeMaxDynamicSharedMemorySize, Traits::kSmemSize));
}
template <class Visit> static decltype(auto) visit_decode_dimension(int dim, Visit &&visit) {
    switch (dim) {
    case 32: return visit(std::integral_constant<int, 32>{});
    case 64: return visit(std::integral_constant<int, 64>{});
    case 96: return visit(std::integral_constant<int, 96>{});
    case 128: return visit(std::integral_constant<int, 128>{});
    default: throw std::invalid_argument("unsupported FlashAttention decode head dimension");
    }
}
template <int Dim, class Element> static int decode_residency() {
    using Traits = Flash_fwd_kernel_traits<Dim, 64, 64, 4, false, false, Element>;
    int blocks = 0;
    CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, flash_attention_paged_decode<Dim, Element, true>,
        Traits::kNThreads, Traits::kSmemSize));
    if (blocks < 1) throw std::runtime_error("split FlashAttention has no resident CTA");
    return blocks;
}
int flash_decode_resident_blocks(int head_dim, Type type) {
    return visit_decode_dimension(head_dim, [&](auto dimension) {
        constexpr int dim = decltype(dimension)::value;
        if (type == Type::F16) return decode_residency<dim, cutlass::half_t>();
        if (type == Type::BF16) return decode_residency<dim, cutlass::bfloat16_t>();
        throw std::invalid_argument("FlashAttention decode requires F16 or BF16");
    });
}
void initialize_flash_attention() {
    configure_dimension<32, cutlass::bfloat16_t>();
    configure_dimension<64, cutlass::bfloat16_t>();
    configure_dimension<96, cutlass::bfloat16_t>();
    configure_dimension<128, cutlass::bfloat16_t>();
    configure_dimension<32, cutlass::half_t>();
    configure_dimension<64, cutlass::half_t>();
    configure_dimension<96, cutlass::half_t>();
    configure_dimension<128, cutlass::half_t>();
    configure<256, false, false>();
    configure<256, false, true>();
    initialize_flash_decode_combine();
}
template <int Dim, bool Causal, bool Local, class Element = cutlass::bfloat16_t>
static void launch(cudaStream_t stream, tts_flash::Flash_fwd_params &p,
                    DenseAttentionLayout dense) {
    using Traits = Flash_fwd_kernel_traits<Dim, 64, 64, 4, false, false, Element>;
    flash_attention_prefill<Dim, Causal, Local, Element>
        <<<dim3((p.seqlen_q + 63) / 64, p.b, p.h), Traits::kNThreads, Traits::kSmemSize, stream>>>(p, dense);
}
template <int Dim, class Element, bool Decode>
static void launch_paged(cudaStream_t stream, tts_flash::Flash_fwd_params &p,
                          DenseAttentionLayout dense, CacheLayout cache, const int *positions) {
    using Traits = Flash_fwd_kernel_traits<Dim, 64, 64, 4, false, false, Element>;
    dim3 grid((p.seqlen_q + 63) / 64, p.b, p.h);
    if constexpr (Decode) {
        if (p.num_splits > 1) {
            flash_attention_paged_decode<Dim, Element, true>
                <<<dim3(1, p.num_splits, p.b * p.h), Traits::kNThreads, Traits::kSmemSize, stream>>>(p, positions, cache);
            combine_flash_decode(stream, p);
        } else {
            flash_attention_paged_decode<Dim, Element>
                <<<grid, Traits::kNThreads, Traits::kSmemSize, stream>>>(p, positions, cache);
        }
    } else if (p.is_causal) {
        flash_attention_paged_prefill<Dim, true, Element>
            <<<grid, Traits::kNThreads, Traits::kSmemSize, stream>>>(p, dense, cache);
    } else {
        flash_attention_paged_prefill<Dim, false, Element>
            <<<grid, Traits::kNThreads, Traits::kSmemSize, stream>>>(p, dense, cache);
    }
}
template <int Dim, bool Decode = false>
static void launch_dimension(cudaStream_t stream, tts_flash::Flash_fwd_params &p,
                               DenseAttentionLayout dense, CacheLayout cache,
                               const int *positions = nullptr) {
    if (cache.valid_paged()) {
        if (p.is_bf16) launch_paged<Dim, cutlass::bfloat16_t, Decode>(stream, p, dense, cache, positions);
        else launch_paged<Dim, cutlass::half_t, Decode>(stream, p, dense, cache, positions);
    } else if constexpr (!Decode) {
        if (p.is_bf16) {
            if (p.is_causal) launch<Dim, true, false>(stream, p, dense);
            else launch<Dim, false, false>(stream, p, dense);
        } else {
            if (p.is_causal) launch<Dim, true, false, cutlass::half_t>(stream, p, dense);
            else launch<Dim, false, false, cutlass::half_t>(stream, p, dense);
        }
    }
}
void prefill_attention(cudaStream_t stream, TensorView q, TensorView k, TensorView v, TensorView out,
                   float *lse, int batch, int time, int slot, int cap, int heads, int kvheads,
                   int dim, bool causal, int window, const SequenceLayout *packed, CacheLayout cache,
                   int key_time, float softmax_scale, DenseAttentionLayout dense) {
    if (key_time && (packed || key_time < 1 || (causal && key_time < time) || key_time > cap))
        throw std::invalid_argument("invalid FlashAttention key length");
    tts_flash::Flash_fwd_params p{};
    p.q_ptr = q.data;
    p.o_ptr = out.data;
    int key_stride = k.data ? k.ld() : kvheads * dim;
    int value_stride = v.data ? v.ld() : kvheads * dim;
    p.k_ptr = k.data ? static_cast<char *>(k.data) + size_t(slot) * cap * key_stride * 2 : nullptr;
    p.v_ptr = v.data ? static_cast<char *>(v.data) + size_t(slot) * cap * value_stride * 2 : nullptr;
    p.q_batch_stride = int64_t(dense.query_batch_rows ? dense.query_batch_rows : time) * q.ld();
    p.o_batch_stride = int64_t(dense.query_batch_rows ? dense.query_batch_rows : time) * out.ld();
    p.k_batch_stride = int64_t(dense.key_batch_rows ? dense.key_batch_rows : cap) * key_stride;
    p.v_batch_stride = int64_t(dense.key_batch_rows ? dense.key_batch_rows : cap) * value_stride;
    p.q_row_stride = q.ld();
    p.o_row_stride = out.ld();
    p.k_row_stride = key_stride;
    p.v_row_stride = value_stride;
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
    p.scale_softmax = softmax_scale > 0 ? softmax_scale : 1.f / std::sqrt(float(dim));
    p.scale_softmax_log2 = p.scale_softmax * 1.4426950408889634f;
    p.p_dropout = p.rp_dropout = 1.f;
    p.p_dropout_in_uint8_t = 255;
    p.scale_softmax_rp_dropout = p.scale_softmax;
    p.is_bf16 = q.type == Type::BF16;
    p.is_causal = causal;
    p.is_seqlens_k_cumulative = true;
    p.window_size_left = window > 0 ? window : -1;
    p.window_size_right = causal ? 0 : (window > 0 ? window : -1);
    p.softmax_lse_ptr = lse;
    if (dense.cu_seqlens_q) {
        p.cu_seqlens_q = const_cast<int *>(dense.cu_seqlens_q);
        p.cu_seqlens_k = const_cast<int *>(dense.cu_seqlens_k);
        p.total_q = q.rows;
        p.unpadded_lse = true;
    }
    if (packed) {
        p.cu_seqlens_q = const_cast<int *>(packed->cu_seqlens);
        p.seqused_k = const_cast<int *>(packed->lengths);
        p.total_q = packed->rows;
        p.unpadded_lse = true;
    }
    if (cache.block_table || cache.sequences || cache.key_pages) {
        if (!cache.valid_paged() || window || (packed && dense.sequences))
            throw std::invalid_argument("unsupported paged FlashAttention layout");
        p.k_ptr = k.data;
        p.v_ptr = v.data;
        p.k_batch_stride = p.v_batch_stride = int64_t(cache.page_size) * kvheads * dim;
        p.block_table = cache.block_table ? const_cast<int *>(cache.block_table) + slot * cache.block_stride : nullptr;
        p.block_table_batch_stride = cache.block_stride;
        p.page_block_size = cache.page_size;
        p.k_pages = cache.key_pages;
        p.v_pages = cache.value_pages;
        p.page_layer_offset = cache.layer * cache.page_layer_bytes / 2;
    }
    if (dim == 32 && !window) launch_dimension<32>(stream, p, dense, cache);
    else if (dim == 64 && !window) launch_dimension<64>(stream, p, dense, cache);
    else if (dim == 96 && !window) launch_dimension<96>(stream, p, dense, cache);
    else if (dim == 128 && !window) launch_dimension<128>(stream, p, dense, cache);
    else if (dim == 128 && causal && q.type == Type::BF16)
        launch<128, true, false>(stream, p, dense);
    else if (dim == 256 && !causal && q.type == Type::BF16) {
        if (window > 0) launch<256, false, true>(stream, p, dense);
        else launch<256, false, false>(stream, p, dense);
    } else throw std::invalid_argument("unsupported FlashAttention specialization");
}
void dense_attention(cudaStream_t stream, TensorView q, TensorView k, TensorView v, TensorView out,
                 float *lse, int batch, int query_time, int key_time, int heads, int dim,
                 bool causal, float scale, DenseAttentionLayout layout) {
    int64_t width = int64_t(heads) * dim;
    int64_t query_rows = int64_t(batch - 1) * (layout.query_batch_rows ? layout.query_batch_rows : query_time) + query_time;
    int64_t key_rows = int64_t(batch - 1) * (layout.key_batch_rows ? layout.key_batch_rows : key_time) + key_time;
    if (batch < 1 || heads < 1 || dim < 1 || width > INT32_MAX || query_time < 1 || key_time < 1 ||
        layout.query_batch_rows < 0 || layout.key_batch_rows < 0 ||
        (layout.query_batch_rows && layout.query_batch_rows < query_time) ||
        (layout.key_batch_rows && layout.key_batch_rows < key_time) ||
        (!layout.sequences && !layout.cu_seqlens_q && (q.rows < query_rows || k.rows < key_rows)) ||
        (bool(layout.cu_seqlens_q) != bool(layout.cu_seqlens_k)) ||
        v.rows != k.rows || out.rows != q.rows || q.cols != width || k.cols != width ||
        v.cols != width || out.cols != width || (q.type != Type::BF16 && q.type != Type::F16) || k.type != q.type ||
        v.type != q.type || out.type != q.type || !lse || !std::isfinite(scale) || scale < 0)
        throw std::invalid_argument("invalid dense FlashAttention views");
    prefill_attention(stream, q, k, v, out, lse, batch, query_time, 0, key_time,
                  heads, heads, dim, causal, 0, nullptr, {}, key_time, scale, layout);
    CUDA(cudaGetLastError());
}
void decode_attention(cudaStream_t stream, TensorView q, TensorView k, TensorView v, TensorView out,
                  float *lse, const int *positions, int batch, int heads, int kvheads,
                  CacheLayout cache, FlashDecodeWorkspace scratch, int partitions) {
    int dim = heads > 0 ? q.cols / heads : 0;
    if (batch < 1 || heads < 1 || kvheads < 1 || heads % kvheads || q.cols != heads * dim ||
        q.rows != batch || out.rows != batch || out.cols != q.cols ||
        (q.type != Type::BF16 && q.type != Type::F16) || out.type != q.type ||
        !cache.valid_paged() || (!positions && !cache.sequences) || !lse ||
        partitions < 1 || partitions > 128 ||
        (partitions > 1 &&
         (int64_t(batch) * heads > 65535 || !scratch.output.data || !scratch.lse.data ||
          scratch.output.type != Type::F32 || scratch.lse.type != Type::F32 ||
          !scratch.output.dense() || !scratch.lse.dense() ||
          scratch.output.numel() < size_t(partitions) * batch * heads * dim ||
          scratch.lse.numel() < size_t(partitions) * batch * heads)))
        throw std::invalid_argument("invalid paged FlashAttention decode layout");
    tts_flash::Flash_fwd_params p{};
    p.q_ptr = q.data;
    p.k_ptr = k.data;
    p.v_ptr = v.data;
    p.o_ptr = out.data;
    p.q_batch_stride = p.q_row_stride = q.ld();
    p.o_batch_stride = p.o_row_stride = out.ld();
    p.k_batch_stride = p.v_batch_stride = int64_t(cache.page_size) * kvheads * dim;
    p.k_row_stride = p.v_row_stride = kvheads * dim;
    p.q_head_stride = p.k_head_stride = p.v_head_stride = p.o_head_stride = dim;
    p.b = batch;
    p.h = heads;
    p.h_k = kvheads;
    p.h_h_k_ratio = heads / kvheads;
    p.d = p.d_rounded = dim;
    p.seqlen_q = 1;
    p.seqlen_q_rounded = 128;
    p.scale_softmax = 1.f / std::sqrt(float(dim));
    p.scale_softmax_log2 = p.scale_softmax * 1.4426950408889634f;
    p.p_dropout = p.rp_dropout = 1.f;
    p.p_dropout_in_uint8_t = 255;
    p.scale_softmax_rp_dropout = p.scale_softmax;
    p.is_bf16 = q.type == Type::BF16;
    p.is_causal = p.is_seqlens_k_cumulative = true;
    p.window_size_left = -1;
    p.window_size_right = 0;
    p.softmax_lse_ptr = lse;
    p.block_table = const_cast<int *>(cache.block_table);
    p.block_table_batch_stride = cache.block_stride;
    p.page_block_size = cache.page_size;
    p.k_pages = cache.key_pages;
    p.v_pages = cache.value_pages;
    p.page_layer_offset = cache.layer * cache.page_layer_bytes / sizeof(cutlass::bfloat16_t);
    p.num_splits = partitions;
    p.oaccum_ptr = scratch.output.data;
    p.softmax_lseaccum_ptr = scratch.lse.data;
    visit_decode_dimension(dim, [&](auto dimension) {
        launch_dimension<decltype(dimension)::value, true>(stream, p, {}, cache, positions);
    });
}
} // namespace tts
