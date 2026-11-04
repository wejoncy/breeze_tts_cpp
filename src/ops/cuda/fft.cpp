#include "runtime/cuda_context.h"
#include "ops/spectral.h"
#include <cufft.h>

namespace tts::ops {
namespace {
void check_fft(cufftResult result) {
    if (result != CUFFT_SUCCESS)
        throw DeviceError("cuFFT failed: " + std::to_string(int(result)));
}
}
void prepare_fft(int device, int fft_size, int rows) {
    auto &cuda = execution_context(device);
    Tensor::Scope allocation(cuda.memory);
    const auto key = std::make_pair(fft_size, rows);
    if (cuda.fft_plans.count(key)) return;
    if (cuda.plans_sealed)
        throw std::logic_error("FFT plan creation after initialization");
    cufftHandle plan = 0;
    check_fft(cufftCreate(&plan));
    try {
        check_fft(cufftSetAutoAllocation(plan, 0));
        size_t workspace_bytes = 0;
        check_fft(cufftMakePlanMany(plan, 1, &fft_size, nullptr, 1, fft_size, nullptr, 1,
                                    fft_size / 2 + 1, CUFFT_R2C, rows, &workspace_bytes));
        check_fft(cufftSetStream(plan, cuda.stream));
        if (workspace_bytes > cuda.fft_workspace.size()) {
            cuda.memory.sync();
            cuda.fft_workspace = Buffer(workspace_bytes);
            for (const auto &entry : cuda.fft_plans)
                check_fft(cufftSetWorkArea(entry.second, cuda.fft_workspace.data()));
        }
        check_fft(cufftSetWorkArea(plan, cuda.fft_workspace.data()));
        cuda.fft_plans.emplace(key, plan);
    } catch (...) {
        cufftDestroy(plan);
        throw;
    }
}
void windowed_fft(TensorView waveform, TensorView window, TensorView frames, TensorView spectrum,
                   int batch, int samples, int time, int hop, int padding) {
    auto &cuda = execution_context(waveform);
    auto plan = cuda.fft_plans.lower_bound({frames.cols, batch * time});
    if (plan == cuda.fft_plans.end() || plan->first.first != frames.cols ||
        plan->first.second > frames.rows)
        throw std::logic_error("missing initialized FFT plan");
    stft_window( waveform, window, frames.slice(0, plan->first.second),
                 batch, samples, time, hop, padding);
    check_fft(cufftExecR2C(plan->second, frames.ptr<float>(), spectrum.ptr<cufftComplex>()));
}
void spectrum_magnitude(TensorView spectrum, TensorView output, float epsilon) {
    tts::spectrum_magnitude( spectrum.data, output, epsilon);
}
void spectral_log(TensorView input, TensorView output, float floor, float scale, float bias) {
    tts::spectral_log( input, output, floor, scale, bias);
}
} // namespace tts::ops
