#pragma once
#include "runtime/device.h"
#include <algorithm>
#include <functional>
#include <vector>
namespace tts::tuning {
struct TimingResources {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    cudaEvent_t start = nullptr, end = nullptr;
    ~TimingResources() {
        if (start)
            cudaEventDestroy(start);
        if (end)
            cudaEventDestroy(end);
        if (executable)
            cudaGraphExecDestroy(executable);
        if (graph)
            cudaGraphDestroy(graph);
    }
};
inline float time_linear(cudaStream_t stream, const std::function<void(int)> &run, int banks) {
    TimingResources t;
    CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    try {
        for (int i = 0; i < banks; ++i)
            run(i);
    } catch (...) {
        cudaStreamEndCapture(stream, &t.graph);
        throw;
    }
    CUDA(cudaStreamEndCapture(stream, &t.graph));
    CUDA(cudaGraphInstantiate(&t.executable, t.graph, nullptr, nullptr, 0));
    CUDA(cudaEventCreate(&t.start));
    CUDA(cudaEventCreate(&t.end));
    for (int i = 0; i < 3; ++i)
        CUDA(cudaGraphLaunch(t.executable, stream));
    std::vector<float> times;
    for (int i = 0; i < 7; ++i) {
        CUDA(cudaEventRecord(t.start, stream));
        CUDA(cudaGraphLaunch(t.executable, stream));
        CUDA(cudaEventRecord(t.end, stream));
        CUDA(cudaEventSynchronize(t.end));
        float ms;
        CUDA(cudaEventElapsedTime(&ms, t.start, t.end));
        times.push_back(ms * 1000 / banks);
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}
} // namespace tts::tuning
