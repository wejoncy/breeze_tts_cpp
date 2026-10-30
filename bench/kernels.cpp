#include "backends/cuda/backend.h"
#include "kernels/attention/flash_attention.h"
#include <cmath>
#include <cuda_bf16.h>
#include <iostream>
#include <chrono>
#include <cstdlib>

// Use a CUDA graph so microsecond kernels are measured without CPU launch gaps.
static int primitive(int argc, char **argv) {
    std::string kind = argv[1];
    int rows = argc > 2 ? std::atoi(argv[2]) : 1;
    int width = argc > 3 ? std::atoi(argv[3]) : 1024;
    int iterations = argc > 4 ? std::atoi(argv[4]) : 1000;
    if (rows < 1 || rows > 512 || iterations < 1)
        throw std::invalid_argument("invalid benchmark size");
    tts::Backend g;
    tts::Buffer x(size_t(rows) * 2048 * 2, &g.memory), u(x.size(), &g.memory), y(x.size(), &g.memory), w(2048 * 2, &g.memory);
    tts::Buffer keys(size_t(rows) * 17 * 2 * 128 * 2, &g.memory), values(keys.size(), &g.memory);
    tts::Buffer lse(size_t(rows) * 8 * sizeof(float), &g.memory);
    for (auto *b : {&x, &u, &y, &w, &keys, &values})
        g.zero(*b);
    if (kind == "--depth-attention" || kind == "--flash-depth-attention") {
        std::vector<__nv_bfloat16> data(keys.size() / 2);
        int seed = 0;
        for (auto *b : {&x, &keys, &values}) {
            ++seed;
            for (size_t i = 0; i < b->size() / 2; ++i)
                data[i] = __nv_bfloat16(std::sin(i * .037f + seed));
            g.upload(b->data(), data.data(), b->size());
            g.sync();
        }
    }
    auto launch = [&] {
        if (kind == "--rms") {
            if (width != 1024 && width != 1152 && width != 2048)
                throw std::invalid_argument("RMS width");
            g.residual_norm(x.view(rows, width), u.view(rows, width), w.view(1, width),
                            y.view(rows, width), 1e-6f);
        } else if (kind == "--depth-attention" || kind == "--flash-depth-attention") {
            if (width < 1 || width > 16)
                throw std::invalid_argument("depth key count");
            if (kind == "--flash-depth-attention")
                tts::flash_prefill(g.stream, x.view(rows, 1024), keys.view(rows * 17, 256),
                                    values.view(rows * 17, 256), y.view(rows, 1024),
                                    lse.ptr<float>(), rows, 1, 0, 17, 8, 2, 128, true, 0,
                                    nullptr, {}, width);
            else
                g.attention(x.view(rows, 1024), keys.view(rows * 17, 256),
                            values.view(rows * 17, 256), y.view(rows, 1024), nullptr, rows, 1, 0,
                            17, 8, 2, 128, true, 0, width - 1);
        } else
            throw std::invalid_argument("unknown primitive");
    };
    launch();
    g.sync();
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    CUDA(cudaStreamBeginCapture(g.stream, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < 100; ++i)
        launch();
    CUDA(cudaStreamEndCapture(g.stream, &graph));
    CUDA(cudaGraphInstantiate(&executable, graph, 0));
    CUDA(cudaGraphLaunch(executable, g.stream));
    g.sync();
    cudaEvent_t a, b;
    CUDA(cudaEventCreate(&a));
    CUDA(cudaEventCreate(&b));
    CUDA(cudaEventRecord(a, g.stream));
    int repeats = (iterations + 99) / 100;
    for (int i = 0; i < repeats; ++i)
        CUDA(cudaGraphLaunch(executable, g.stream));
    CUDA(cudaEventRecord(b, g.stream));
    CUDA(cudaEventSynchronize(b));
    float ms;
    CUDA(cudaEventElapsedTime(&ms, a, b));
    std::cout << "{\"primitive\":\"" << kind << "\",\"rows\":" << rows
              << ",\"width_or_keys\":" << width << ",\"us\":" << ms * 10 / repeats << "}\n";
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    cudaGraphExecDestroy(executable);
    cudaGraphDestroy(graph);
    return 0;
}
int main(int argc, char **argv) {
    try {
        if (argc > 1 && argv[1][0] == '-')
            return primitive(argc, argv);
        int n = argc > 1 ? std::atoi(argv[1]) : 16384, k = argc > 2 ? std::atoi(argv[2]) : 1024,
            m = argc > 3 ? std::atoi(argv[3]) : 1;
        int iterations = argc > 4 ? std::atoi(argv[4]) : 100;
        if (n < 1 || k < 1 || m < 1 || iterations < 1)
            throw std::runtime_error("positive dimensions required");
        tts::Backend g;
        std::vector<tts::Buffer> weights;
        for (int i = 0; i < 8; i++) {
            weights.emplace_back(size_t(n) * k * 2, &g.memory);
            g.zero(weights.back());
        }
        tts::Buffer x(size_t(m) * k * 2, &g.memory), y(size_t(m) * n * 2, &g.memory);
        g.zero(x);
        for (int i = 0; i < 8; i++)
            g.linear(x.view(m, k), weights[i].view(n, k), y.view(m, n));
        g.sync();
        cudaEvent_t a, b;
        CUDA(cudaEventCreate(&a));
        CUDA(cudaEventCreate(&b));
        CUDA(cudaEventRecord(a, g.stream));
        for (int i = 0; i < iterations; i++)
            g.linear(x.view(m, k), weights[i % 8].view(n, k), y.view(m, n));
        CUDA(cudaEventRecord(b, g.stream));
        CUDA(cudaEventSynchronize(b));
        float ms;
        CUDA(cudaEventElapsedTime(&ms, a, b));
        double us = ms * 1000 / iterations,
               gbps = (double(n) * k * 2 + double(m) * (k + n) * 2) / (us * 1000),
               tflops = 2. * m * n * k / (us * 1e6);
        std::cout << "{\"implementation\":\""
                  << (g.custom_gemv && m <= 2 && k <= 1024 && n <= 4096 ? "custom" : "cublas")
                  << "\",\"m\":" << m << ",\"n\":" << n << ",\"k\":" << k << ",\"us\":" << us
                  << ",\"effective_GBps\":" << gbps << ",\"TFLOPS\":" << tflops << "}\n";
        cudaEventDestroy(a);
        cudaEventDestroy(b);
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
