#include "runtime/cuda_context.h"
#include "runtime/memory_log.h"
#include "support/init_log.h"
#include "support/log.h"
#include "ops/cuda/blas_check.h"
#include "ops/cuda/lt_linear.h"
#include "ops/cuda/quantization.h"
#if TTS_HAVE_CUFFT
#include <cufft.h>
#endif

namespace tts {
namespace {
constexpr double mib = 1024.0 * 1024.0;
std::shared_ptr<DeviceContext> create_memory(int device, size_t limit) {
    log::InitPhase phase("CUDA context and memory pool");
    CUDA(cudaSetDevice(device));
    cudaDeviceProp properties{};
    CUDA(cudaGetDeviceProperties(&properties, device));
    TTS_LOG(TTS_LOG_INFO, "engine.init", "device={} name=\"{}\" sm={}.{} multiprocessors={}",
            device, properties.name, properties.major, properties.minor,
            properties.multiProcessorCount);
    auto memory = std::make_shared<DeviceContext>(device, limit);
    const size_t total = log_device_memory(*memory, "CUDA context ready");
    TTS_LOG(limit ? TTS_LOG_INFO : TTS_LOG_DEBUG, "engine.init",
            "engine_limit={:.0f} MiB limit_mode={} (CUDA pool allocations)",
            (limit ? limit : total) / mib, limit ? "configured" : "device capacity");
    return memory;
}
} // namespace
size_t log_device_memory(const DeviceContext &memory, const char *phase) {
    // Initialization already owns the current device. No stream synchronization is added.
    size_t available = 0, total = 0;
    CUDA(cudaMemGetInfo(&available, &total));
    if (log::enabled(TTS_LOG_INFO)) {
        const auto pool = memory.stats();
        TTS_LOG(TTS_LOG_INFO, "engine.memory",
                "phase={} device={} gpu_total={:.0f} MiB gpu_free={:.0f} MiB gpu_used={:.0f} MiB "
                "pool_used={:.0f} MiB pool_reserved={:.0f} MiB pool_peak_reserved={:.0f} MiB",
                phase, memory.device, total / mib, available / mib, (total - available) / mib,
                pool.used_bytes / mib, pool.reserved_bytes / mib, pool.peak_reserved_bytes / mib);
    }
    return total;
}
CudaContext::CudaContext(int ordinal, Quantization format, size_t limit)
    : memory_owner(create_memory(ordinal, limit)), memory(*memory_owner),
      device(ordinal), stream(memory.stream) {
    ExecutionScope execution(*this);
    Tensor::Scope allocations(memory);
    try {
        log::InitPhase phase("CUDA libraries");
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
        phase.next("CUDA kernel initialization");
        initialize_cuda_kernels(*this);
        if (format != Quantization::None) {
            phase.next("quantized GEMM initialization");
            quantization = std::make_unique<QuantizedGemm>(stream, lt, gemm_workspace, quantized_type(format));
        }
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
