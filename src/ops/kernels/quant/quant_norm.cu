#include "ops/kernels/quant/quant_norm.h"
#include "ops/kernels/quant/quant_norm_kernels.cuh"

namespace tts {
namespace {
using bf16 = __nv_bfloat16;
template <Type Storage, class T>
void launch(T *x, const T *u, const T *w, const T *scale, unsigned char *q, float *s, int rows,
            int width, float eps, bool gemma, cudaStream_t stream) {
    if (u)
        norm_kernels::residual_rms_quantize<256, 8, true, Storage, T>
            <<<rows, 256, 0, stream>>>(x, u, w, scale, q, s, width, eps, gemma);
    else
        norm_kernels::residual_rms_quantize<256, 8, false, Storage, T>
            <<<rows, 256, 0, stream>>>(x, u, w, scale, q, s, width, eps, gemma);
}
template <class T>
void launch_format(Type format, TensorView x, TensorView u, TensorView w, TensorView scale,
                   RowScaledView out, float eps, bool gemma, cudaStream_t stream) {
    if (format == Type::I8)
        launch<Type::I8>(x.ptr<T>(), u.ptr<T>(), w.ptr<T>(), scale.ptr<T>(),
                         out.values.ptr<unsigned char>(), out.scales.ptr<float>(), x.rows, x.cols,
                         eps, gemma, stream);
    else
        launch<Type::FP8>(x.ptr<T>(), u.ptr<T>(), w.ptr<T>(), scale.ptr<T>(),
                          out.values.ptr<unsigned char>(), out.scales.ptr<float>(), x.rows, x.cols,
                          eps, gemma, stream);
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
    if (out.values.type != impl->format || (x.type != Type::BF16 && x.type != Type::F16) ||
        !x.dense() || (x.cols < 1 || x.cols > 2048 || x.rows < 1) || x.cols != out.values.cols ||
        x.rows != out.values.rows || w.type != x.type || !w.dense() ||
        w.numel() != size_t(x.cols) ||
        (u.data && (u.type != x.type || u.rows != x.rows || u.cols != x.cols || !u.dense())) ||
        (scale.data && (scale.type != x.type || !scale.dense() || scale.numel() != size_t(x.cols))))
        throw std::invalid_argument("quantized RMS shape/type mismatch");
    if (x.type == Type::F16)
        launch_format<__half>(impl->format, x, u, w, scale, out, eps, gemma, impl->stream);
    else
        launch_format<bf16>(impl->format, x, u, w, scale, out, eps, gemma, impl->stream);
}
} // namespace tts
