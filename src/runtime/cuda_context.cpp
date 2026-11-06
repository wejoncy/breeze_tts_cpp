#include "runtime/cuda_context.h"
#include "ops/cuda/blas_check.h"
#include "ops/cuda/lt_linear.h"
#include "ops/cuda/quantization.h"
#if TTS_HAVE_CUFFT
#include <cufft.h>
#endif

namespace tts {
CudaContext::CudaContext(int ordinal, Quantization format, size_t limit)
    : memory_owner(std::make_shared<DeviceContext>(ordinal, limit)), memory(*memory_owner),
      device(ordinal), stream(memory.stream) {
    ExecutionScope execution(*this);
    Tensor::Scope allocations(memory);
    try {
        CUDA(cudaSetDevice(device));
        cudaDeviceProp properties{};
        CUDA(cudaGetDeviceProperties(&properties, device));
        if (properties.major < 8) throw std::invalid_argument("CUDA inference requires SM80+");
        shared_memory_per_block = properties.sharedMemPerBlock;
        multiprocessors = properties.multiProcessorCount;
        resident_threads = multiprocessors * properties.maxThreadsPerMultiProcessor;
        sm_major = properties.major;
        sm_minor = properties.minor;
        gemm_workspace = Buffer(8 << 20);
        BLAS(cublasCreate(&blas));
        BLAS(cublasSetStream(blas, stream));
        BLAS(cublasSetWorkspace(blas, gemm_workspace.data(), gemm_workspace.size()));
        BLAS(cublasSetMathMode(blas, CUBLAS_TF32_TENSOR_OP_MATH));
        BLAS(cublasLtCreate(&lt));
        matmul = std::make_unique<LtLinear>(lt, blas);
        initialize_cuda_kernels(*this);
        if (format != Quantization::None)
            quantization = std::make_unique<QuantizedGemm>(memory, lt, gemm_workspace, quantized_type(format));
    } catch (...) {
        quantization.reset(); matmul.reset();
        if (lt) cublasLtDestroy(lt);
        if (blas) cublasDestroy(blas);
        throw;
    }
}
CudaContext::~CudaContext() {
    cudaStreamSynchronize(stream);
#if TTS_HAVE_CUFFT
    for (const auto &plan : fft_plans) cufftDestroy(plan.second);
#endif
    quantization.reset(); matmul.reset();
    if (lt) cublasLtDestroy(lt);
    if (blas) cublasDestroy(blas);
}
} // namespace tts
