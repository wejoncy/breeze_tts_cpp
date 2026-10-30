#pragma once
#include "runtime/device.h"
#include "core/kv_cache_view.h"
#include "core/packed_batch.h"
#include "kernels/quant/quant_backend.h"
#include "backends/cuda/blas_check.h"
#include <nvtx3/nvToolsExt.h>
#include <vector>
namespace tts {
struct Range {
    explicit Range(const char *n) {
        nvtxRangePushA(n);
    }
    ~Range() {
        nvtxRangePop();
    }
};
class Backend {
  public:
    DeviceContext memory;
    const int device;
    const cudaStream_t stream;

  private:
    Buffer blas_workspace_;
    Buffer flash_lse_;
    std::unique_ptr<QuantBackend> quant_;

  public:
    cublasHandle_t blas = nullptr;
    bool custom_gemv = true;
    bool pdl = false;        // Experimental SM90 RMS -> narrow GEMV overlap; opt-in only.
    bool force_gemv = false; // Benchmark-only legacy baseline, off by default.
    explicit Backend(int device = 0, Quantization quantization = Quantization::None,
                     size_t memory_limit_bytes = 0);
    ~Backend();
    Backend(const Backend &) = delete;
    void begin_admission() {
        if (quant_)
            quant_->begin_admission();
    }
    void seal_plans() {
        if (quant_)
            quant_->seal();
    }
    QuantBackend &quantized() {
        if (!quant_)
            throw std::logic_error("FP8 backend is not enabled");
        return *quant_;
    }
    void sync() {
        CUDA(cudaStreamSynchronize(stream));
    }
    // Explicit preparation only. Caller invalidates graphs before changing this storage.
    void reserve_attention_workspace(size_t elements) {
        if (elements > SIZE_MAX / sizeof(float))
            throw std::overflow_error("attention workspace size");
        if (elements * sizeof(float) > flash_lse_.size()) {
            sync();
            flash_lse_ = Buffer(elements * sizeof(float), &memory);
        }
    }
    void zero(const Buffer &b) {
        CUDA(cudaMemsetAsync(b.data(), 0, b.size(), stream));
    }
    void upload(void *dst, const void *src, size_t n) {
        CUDA(cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, stream));
    }
    void download(void *dst, const void *src, size_t n) {
        CUDA(cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToHost, stream));
        sync();
    }
    void linear(TensorView x, TensorView w, TensorView y, bool full_precision = false);
    void norm(TensorView x, TensorView w, TensorView y, float eps, bool gemma = false,
              TensorView bias = {});
    void residual_norm(TensorView residual, TensorView update, TensorView w, TensorView normalized,
                       float eps, TensorView scale = {}, bool gemma = false);
    void residual_norm_last(TensorView residual, TensorView update, TensorView w, TensorView out,
                            const int *ends, float eps);
    void add(TensorView a, TensorView b, TensorView out, TensorView scale = {});
    void bias(TensorView x, TensorView b);
    void activation(TensorView x, TensorView y, int kind); // 0 SiLU, 1 tanh GELU, 2 erf GELU
    void gated(TensorView gu, TensorView y, int kind);
    void embedding(TensorView table, const int *ids, TensorView out, float scale = 1,
                   TensorView replacement = {}, int replacement_id = -1);
    void make_rope(TensorView table, int dim, float theta, float factor, int rope_type,
                   float low_wave = 0, float high_wave = 0);
    void qkv(TensorView in, TensorView q, TensorView cache_k, TensorView cache_v,
             const int *positions, int batch, int time, int slot, int capacity, int heads,
             int kvheads, int dim, TensorView qnorm, TensorView knorm, float eps, bool gemma,
             TensorView rope, int fixed_position = -1, const PackedBatch *packed = nullptr,
             CacheLayout cache = {});
    void attention(TensorView q, TensorView k, TensorView v, TensorView out, const int *positions,
                   int batch, int time, int slot, int capacity, int heads, int kvheads, int dim,
                   bool causal, int window, int fixed_position = -1,
                   const PackedBatch *packed = nullptr, CacheLayout cache = {});
    void increment(int *positions, int n, int value = 1, const CacheSequence *sequences = nullptr);
};
} // namespace tts
