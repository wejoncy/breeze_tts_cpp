#pragma once
#include <cuda_runtime_api.h>
#include <chrono>
#include <cstddef>
#include <functional>

namespace tts {
// Init diagnostics shared by graph families; no state or logging in replay.
class GraphCaptureProgress {
    const char *name_;
    size_t total_, done_ = 0;
    std::chrono::steady_clock::time_point start_ = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point next_report_ = start_ + std::chrono::milliseconds(200);
    void report() const;

  public:
    GraphCaptureProgress(const char *name, size_t total);
    void advance();
};
struct Graph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    Graph() = default;
    Graph(const Graph &) = delete;
    Graph &operator=(const Graph &) = delete;
    ~Graph();
    void capture(cudaStream_t stream, const std::function<void()> &forward);
    void run(cudaStream_t stream) const;
};
} // namespace tts
