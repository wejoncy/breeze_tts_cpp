#include "kernels/quant/quant_norm.h"
#include "kernels/quant/quant_norm_kernels.cuh"

namespace tts {
namespace {
using bf16 = __nv_bfloat16;
template <Type Storage>
void launch(bf16 *x, const bf16 *u, const bf16 *w, const bf16 *scale, unsigned char *q, float *s,
            int rows, int width, float eps, bool gemma, cudaStream_t stream) {
    if (u)
        norm_kernels::residual_rms_quantize<256, 8, true, Storage>
            <<<rows, 256, 0, stream>>>(x, u, w, scale, q, s, width, eps, gemma);
    else
        norm_kernels::residual_rms_quantize<256, 8, false, Storage>
            <<<rows, 256, 0, stream>>>(x, u, w, scale, q, s, width, eps, gemma);
}
} // namespace
struct QuantNorm::Impl {
    Type format;
    cudaStream_t stream;
    Impl(DeviceContext &memory, Type f) : format(f), stream(memory.stream) {}
};
QuantNorm::QuantNorm(DeviceContext &memory, Type format)
    : impl(std::make_unique<Impl>(memory, format)) {}
QuantNorm::~QuantNorm() = default;
void QuantNorm::run(TensorView x, TensorView u, TensorView w, RowScaledView out, float eps,
                    bool gemma, TensorView scale) {
    out.validate();
    if (out.values.type != impl->format || x.type != Type::BF16 || !x.dense() ||
        (x.cols < 1 || x.cols > 2048 || x.rows < 1) || x.cols != out.values.cols ||
        x.rows != out.values.rows || w.type != Type::BF16 || !w.dense() ||
        w.numel() != size_t(x.cols) ||
        (u.data && (u.type != x.type || u.rows != x.rows || u.cols != x.cols || !u.dense())) ||
        (scale.data &&
         (scale.type != Type::BF16 || !scale.dense() || scale.numel() != size_t(x.cols))))
        throw std::invalid_argument("quantized RMS shape/type mismatch");
    if (impl->format == Type::I8)
        launch<Type::I8>(x.ptr<bf16>(), u.ptr<bf16>(), w.ptr<bf16>(), scale.ptr<bf16>(),
                         out.values.ptr<unsigned char>(), out.scales.ptr<float>(), x.rows, x.cols,
                         eps, gemma, impl->stream);
    else
        launch<Type::FP8>(x.ptr<bf16>(), u.ptr<bf16>(), w.ptr<bf16>(), scale.ptr<bf16>(),
                          out.values.ptr<unsigned char>(), out.scales.ptr<float>(), x.rows, x.cols,
                          eps, gemma, impl->stream);
}
} // namespace tts
