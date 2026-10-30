#pragma once
#include "runtime/device.h"
#include <functional>

namespace tts {
struct Graph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    Graph() = default;
    Graph(const Graph &) = delete;
    Graph &operator=(const Graph &) = delete;
    ~Graph() {
        if (executable)
            cudaGraphExecDestroy(executable);
        if (graph)
            cudaGraphDestroy(graph);
    }
    void capture(cudaStream_t stream, const std::function<void()> &fn) {
        if (executable)
            throw std::logic_error("graph already captured");
        if (graph) {
            CUDA(cudaGraphDestroy(graph));
            graph = nullptr;
        }
        CUDA(cudaStreamSynchronize(stream));
        CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        try {
            fn();
        } catch (...) {
            cudaStreamEndCapture(stream, &graph);
            throw;
        }
        CUDA(cudaStreamEndCapture(stream, &graph));
        CUDA(cudaGraphInstantiate(&executable, graph, 0));
        CUDA(cudaGraphUpload(executable, stream));
        CUDA(cudaStreamSynchronize(stream));
    }
    void run(cudaStream_t stream) {
        CUDA(cudaGraphLaunch(executable, stream));
    }
};
} // namespace tts
