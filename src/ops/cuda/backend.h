#pragma once
#include "runtime/device.h"
#include "ops/cuda/execution_context.h"
#include "core/kv_cache_view.h"
#include "core/qkv_projection.h"
#include "core/sequence_layout.h"
#include "ops/attention.h"
#include "ops/cuda/quantization.h"
#include "ops/relative_attention.h"
#include "ops/spectral.h"
#include "support/profiler.h"
#include <vector>
#include <array>
#include <climits>
#include <map>
namespace tts {
class LtLinear;
class Backend {
  public:
    DeviceContext memory;
    const int device;
    const cudaStream_t stream;

  private:
    Buffer blas_workspace_;
    Buffer flash_lse_, flash_decode_output_, flash_decode_lse_;
    Buffer fft_workspace_;
    std::map<std::pair<int, int>, int> fft_plans_;
    friend void ops::prepare_fft(int, int, int);
    friend void ops::windowed_fft(TensorView, TensorView, TensorView, TensorView,
                                  int, int, int, int, int);
    int qkv_gemv_capacity_ = 0;
    std::array<int, 4> gemv_slots_{};
    bool pdl_ = false;
    cublasLtHandle_t lt_ = nullptr;
    std::unique_ptr<LtLinear> lt_linear_;
    std::unique_ptr<QuantBackend> quant_;
    size_t shared_memory_per_block_ = 0;
    int multiprocessors_ = 0, normalized_linear_capacity_ = 0, normalized_qkv_capacity_ = 0;
    cublasHandle_t blas = nullptr;
    friend void ops::relative_attention(TensorView, TensorView, TensorView,
                                        TensorView, TensorView, TensorView, int, int, int, int,
                                        const SequenceLayout *);

    std::array<int, 8> flash_decode_slots_{};
    bool plans_sealed_ = false;

  public:
    explicit Backend(int device = 0, Quantization quantization = Quantization::None,
                     size_t memory_limit_bytes = 0);
    ~Backend();
    Backend(const Backend &) = delete;
    size_t shared_memory_per_block() const {
        return shared_memory_per_block_;
    }
    int multiprocessors() const { return multiprocessors_; }
    int normalized_linear_capacity(bool qkv_writeback = false) const {
        return qkv_writeback ? normalized_qkv_capacity_ : normalized_linear_capacity_;
    }
    int flash_decode_capacity(int head_dim, Type type) const {
        if ((type != Type::F16 && type != Type::BF16) || head_dim < 32 ||
            head_dim > 128 || head_dim % 32) return 0;
        return flash_decode_slots_[(type == Type::F16 ? 4 : 0) + head_dim / 32 - 1];
    }
    void reserve_flash_decode_workspace(int head_dim, Type type) {
        const int slots = flash_decode_capacity(head_dim, type);
        if (!slots) return; // Unsupported geometry retains its existing attention path.
        const size_t elements = size_t(slots) * head_dim;
        if (elements > INT_MAX || size_t(slots) > SIZE_MAX / sizeof(float))
            throw std::overflow_error("decode workspace exceeds tensor capacity");
        const size_t output_bytes = elements * sizeof(float), lse_bytes = size_t(slots) * sizeof(float);
        if (output_bytes <= flash_decode_output_.size() && lse_bytes <= flash_decode_lse_.size()) return;
        if (plans_sealed_) throw std::logic_error("decode workspace growth after initialization");
        sync();
        if (output_bytes > flash_decode_output_.size())
            flash_decode_output_ = Buffer(output_bytes, &memory);
        if (lse_bytes > flash_decode_lse_.size())
            flash_decode_lse_ = Buffer(lse_bytes, &memory);
    }
    FlashDecodeWorkspace flash_decode_workspace() const {
        if (!flash_decode_output_.size()) return {};
        return {flash_decode_output_.view(1, int(flash_decode_output_.size() / sizeof(float)), Type::F32),
                flash_decode_lse_.view(1, int(flash_decode_lse_.size() / sizeof(float)), Type::F32)};
    }
    void seal_plans();
    bool has_quantization() const { return bool(quant_); }
    QuantBackend &quantized() {
        if (!quant_)
            throw std::logic_error("FP8 backend is not enabled");
        return *quant_;
    }
    void sync() {
        memory.sync();
    }
    // Explicit preparation only. Caller invalidates graphs before changing this storage.
    void reserve_attention_workspace(size_t elements) {
        if (elements > SIZE_MAX / sizeof(float))
            throw std::overflow_error("attention workspace size");
        if (elements * sizeof(float) > flash_lse_.size()) {
            if (plans_sealed_)
                throw std::logic_error("attention workspace growth after initialization");
            sync();
            flash_lse_ = Buffer(elements * sizeof(float), &memory);
        }
    }
    void zero(const Buffer &b) {
        CUDA(cudaMemsetAsync(b.data(), 0, b.size(), stream));
    }
    void upload(void *dst, const void *src, size_t n) {
        memory.upload(dst, src, n);
    }
    void download(void *dst, const void *src, size_t n) {
        memory.download(dst, src, n);
    }
    void linear(TensorView x, TensorView w, TensorView y, bool full_precision = false,
                float accumulation = 0, float product_scale = 1);
    void grouped_linear(TensorView x, TensorView weight, TensorView output, int groups);
    void prepare_linear(TensorView x, TensorView w, TensorView y, bool full_precision = false,
                        float accumulation = 0, float product_scale = 1, int groups = 1, int max_rows = 0);
    bool prepare_linear_gelu(TensorView x, TensorView w, TensorView bias, TensorView y,
                             int max_rows);
    bool linear_gelu(TensorView x, TensorView w, TensorView bias, TensorView y);
    bool linear_qkv(TensorView x, TensorView w, const QkvProjectionOutput &);
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
             TensorView rope, int fixed_position = -1, const SequenceLayout *packed = nullptr,
             CacheLayout cache = {}, TensorView bias = {});
    void attention(TensorView q, TensorView k, TensorView v, TensorView out, const int *positions,
                   int batch, int time, int slot, int capacity, int heads, int kvheads, int dim,
                   bool causal, int window, int fixed_position = -1,
                   const SequenceLayout *packed = nullptr, CacheLayout cache = {});
    void increment(int *positions, int n, int value = 1, const CacheSequence *sequences = nullptr);
};
} // namespace tts
