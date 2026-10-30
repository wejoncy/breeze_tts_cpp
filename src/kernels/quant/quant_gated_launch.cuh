#pragma once
#include "kernels/quant/quant_gated_kernels.cuh"
#include "runtime/device.h"
namespace tts::gated_dispatch {
namespace {
using bf16 = __nv_bfloat16;
// Keep host launch wrappers local to the CUDA module they were compiled with.
// Otherwise another executable/DSO can interpose a different-architecture image.
inline bool supports_clusters(int device) {
    int supported = 0;
    CUDA(cudaDeviceGetAttribute(&supported, cudaDevAttrClusterLaunch, device));
    if (!supported)
        return false;
    cudaFuncAttributes compiled{};
    CUDA(cudaFuncGetAttributes(
        &compiled, gated_kernels::vector_gated_quantize<0, 256, 4, false, 2, Type::I8>));
    // PTX compiled below SM90 omits the cross-CTA reduction, even when JITed on Hopper.
    return compiled.ptxVersion >= 90;
}
template <int Kind, int T, int V, int Parts, bool Table, Type Storage>
void run(const bf16 *x, unsigned char *q, float *s, int rows, int width, const bf16 *table,
         cudaStream_t stream) {
    if constexpr (Parts == 1)
        gated_kernels::vector_gated_quantize<Kind, T, V, Table, 1, Storage>
            <<<rows, T, 0, stream>>>(x, q, s, width, table);
    else {
        cudaLaunchConfig_t cfg{};
        cfg.gridDim = dim3(rows * Parts);
        cfg.blockDim = dim3(T);
        cfg.stream = stream;
        cudaLaunchAttribute attr{};
        attr.id = cudaLaunchAttributeClusterDimension;
        attr.val.clusterDim = {Parts, 1, 1};
        cfg.attrs = &attr;
        cfg.numAttrs = 1;
        CUDA(cudaLaunchKernelEx(
            &cfg, gated_kernels::vector_gated_quantize<Kind, T, V, Table, Parts, Storage>, x, q, s,
            width, table));
    }
}
} // namespace
} // namespace tts::gated_dispatch
