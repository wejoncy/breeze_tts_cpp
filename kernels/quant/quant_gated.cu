#include "kernels/quant/quant_gated.h"
#include "kernels/quant/quant_gated_launch.cuh"
#include <algorithm>
namespace tts {
using bf16 = __nv_bfloat16;
namespace {
template <Type Storage, int Kind>
void launch(TensorView input, RowScaledView out, int parts, cudaStream_t stream) {
#define RUN(P)                                                                                     \
    case P:                                                                                        \
        gated_dispatch::run<Kind, 256, P == 1 ? 8 : 4, P, false, Storage>(                          \
            input.ptr<bf16>(), out.values.ptr<unsigned char>(), out.scales.ptr<float>(),           \
            input.rows, out.values.cols, nullptr, stream);                                         \
        break
    switch (parts) {
        RUN(1);
        RUN(2);
        RUN(4);
        RUN(8);
    }
#undef RUN
}
} // namespace
struct QuantGated::Impl {
    Type format;
    cudaStream_t stream;
    int sms;
    bool clusters;
    Impl(DeviceContext &memory, Type f) : format(f), stream(memory.stream) {
        cudaDeviceProp property{};
        CUDA(cudaGetDeviceProperties(&property, memory.device));
        sms = property.multiProcessorCount;
        clusters = gated_dispatch::supports_clusters(memory.device);
    }
};
QuantGated::QuantGated(DeviceContext &memory, Type format)
    : impl(std::make_unique<Impl>(memory, format)) {}
QuantGated::~QuantGated() = default;
void QuantGated::run(TensorView input, RowScaledView out, int kind) {
    out.validate();
    int width = out.values.cols;
    if (out.values.type != impl->format || input.type != Type::BF16 || !input.dense() ||
        input.rows != out.values.rows || input.cols != 2 * width || kind < 0 || kind > 1 ||
        width < 1 || width > 8192 || input.rows < 1)
        throw std::invalid_argument("quantized gated shape/type mismatch");
    int parts = 1;
    // Independent CTAs use 128-bit input loads. Clusters use 64-bit packs so
    // that short batches can spread the same token across more SMs.
    if (impl->clusters)
        while (parts < 8 && parts * 2 <= impl->sms / input.rows && 256 * 4 * parts < width)
            parts *= 2;
    if (impl->format == Type::I8) {
        if (kind == 0)
            launch<Type::I8, 0>(input, out, parts, impl->stream);
        else
            launch<Type::I8, 1>(input, out, parts, impl->stream);
    } else {
        if (kind == 0)
            launch<Type::FP8, 0>(input, out, parts, impl->stream);
        else
            launch<Type::FP8, 1>(input, out, parts, impl->stream);
    }
}
} // namespace tts
