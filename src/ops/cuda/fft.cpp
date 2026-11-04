#include "ops/cuda/backend.h"
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
    auto &backend = execution_context(device);
    const auto key = std::make_pair(fft_size, rows);
    if (backend.fft_plans_.count(key)) return;
    if (backend.plans_sealed_)
        throw std::logic_error("FFT plan creation after initialization");
    cufftHandle plan = 0;
    check_fft(cufftCreate(&plan));
    try {
        check_fft(cufftSetAutoAllocation(plan, 0));
        size_t workspace_bytes = 0;
        check_fft(cufftMakePlanMany(plan, 1, &fft_size, nullptr, 1, fft_size, nullptr, 1,
                                    fft_size / 2 + 1, CUFFT_R2C, rows, &workspace_bytes));
        check_fft(cufftSetStream(plan, backend.stream));
        if (workspace_bytes > backend.fft_workspace_.size()) {
            backend.sync();
            backend.fft_workspace_ = Buffer(workspace_bytes, &backend.memory);
            for (const auto &entry : backend.fft_plans_)
                check_fft(cufftSetWorkArea(entry.second, backend.fft_workspace_.data()));
        }
        check_fft(cufftSetWorkArea(plan, backend.fft_workspace_.data()));
        backend.fft_plans_.emplace(key, plan);
    } catch (...) {
        cufftDestroy(plan);
        throw;
    }
}
void windowed_fft(TensorView waveform, TensorView window, TensorView frames, TensorView spectrum,
                   int batch, int samples, int time, int hop, int padding) {
    auto &backend = execution_context(waveform);
    auto plan = backend.fft_plans_.lower_bound({frames.cols, batch * time});
    if (plan == backend.fft_plans_.end() || plan->first.first != frames.cols ||
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
