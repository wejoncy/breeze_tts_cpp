#pragma once
#include "runtime/device.h"
#include "support/log.h"
#include <chrono>
#include <functional>

namespace tts {
// Init diagnostics shared by graph families; no state or logging in replay.
class GraphCaptureProgress {
    const char *name_;
    size_t total_, done_ = 0;
    std::chrono::steady_clock::time_point start_ = std::chrono::steady_clock::now();
    void report() const {
        size_t filled = total_ ? 20 * done_ / total_ : 20;
        TTS_LOG(TTS_LOG_INFO, "engine.init",
                "CUDA graph capture {} [{}{}] {}/{} elapsed_ms={:.0f}", name_,
                std::string(filled, '='), std::string(20 - filled, ' '), done_, total_,
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start_).count());
    }
  public:
    GraphCaptureProgress(const char *name, size_t total) : name_(name), total_(total) { report(); }
    void advance() { ++done_; report(); }
};
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
    void run(cudaStream_t stream) const {
        CUDA(cudaGraphLaunch(executable, stream));
    }
};
} // namespace tts
