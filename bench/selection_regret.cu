// Compare the analytical production choice against native candidates at actual M.
// The production choice may also use cuBLASLt; the reference here is native only.
#include "backends/cuda/backend.h"
#include "kernels/quant/quant_kernel_catalog.h"
#include "kernels/quant/quant_dispatch.h"
#include "runtime/graph.h"
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
using namespace tts;
using bf16 = __nv_bfloat16;
__global__ void values(bf16 *x, size_t count) {
    size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (i < count)
        x[i] = bf16(float(int((i * 1664525u + 1013904223u) % 65536) - 32768) / 32768.f);
}
float time_path(Backend &g, const std::function<void(int)> &run, int banks) {
    Graph graph;
    graph.capture(g.stream, [&] {
        for (int b = 0; b < banks; ++b)
            run(b);
    });
    for (int i = 0; i < 4; ++i)
        graph.run(g.stream);
    g.sync();
    cudaEvent_t a, b;
    CUDA(cudaEventCreate(&a));
    CUDA(cudaEventCreate(&b));
    std::vector<float> times;
    for (int i = 0; i < 9; ++i) {
        CUDA(cudaEventRecord(a, g.stream));
        graph.run(g.stream);
        CUDA(cudaEventRecord(b, g.stream));
        CUDA(cudaEventSynchronize(b));
        float ms;
        CUDA(cudaEventElapsedTime(&ms, a, b));
        times.push_back(ms * 1000 / banks);
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    std::sort(times.begin(), times.end());
    return times[4];
}
int main() {
    Backend g(0, Quantization::FP8);
    cudaDeviceProp property{};
    CUDA(cudaGetDeviceProperties(&property, 0));
    Buffer scratch(8 << 20, &g.memory);
    for (auto shape : {std::pair{1536, 1024}, std::pair{1152, 6912}}) {
        int n = shape.first, k = shape.second;
        for (int m : {3, 7, 17, 31, 33, 65, 129, 257}) {
            size_t matrix = size_t(n) * k;
            int banks =
                int(std::clamp<size_t>((2 * property.l2CacheSize + matrix - 1) / matrix, 4, 128));
            Buffer x(size_t(m) * k * 2, &g.memory), w(matrix * banks * 2, &g.memory), qx(size_t(m) * k, &g.memory),
                qw(matrix * banks, &g.memory), sx(m * 4, &g.memory), sw(size_t(banks) * n * 4, &g.memory), y(size_t(m) * n * 2, &g.memory);
            values<<<(x.size() / 2 + 255) / 256, 256, 0, g.stream>>>(x.ptr<bf16>(), x.size() / 2);
            values<<<(w.size() / 2 + 255) / 256, 256, 0, g.stream>>>(w.ptr<bf16>(), w.size() / 2);
            auto &q = g.quantized();
            RowScaledView a{qx.view(m, k, Type::FP8), sx.view(m, 1, Type::F32)};
            q.quantize(x.view(m, k), a);
            q.quantize(w.view(n * banks, k),
                       {qw.view(n * banks, k, Type::FP8), sw.view(n * banks, 1, Type::F32)});
            g.begin_admission();
            q.prepare(m, n, k);
            auto weight = [&](int bank) {
                return RowScaledView{qw.view(n, k, Type::FP8, matrix * bank),
                                     sw.view(n, 1, Type::F32, size_t(bank) * n * 4)};
            };
            float selected =
                time_path(g, [&](int bank) { q.linear(a, weight(bank), y.view(m, n)); }, banks);
            float best = INFINITY;
            std::string best_name;
            for (auto config : quant_linear_candidates(
                     {m, n, k}, {true, size_t(property.sharedMemPerBlock), scratch.size()})) {
                if (config.kind == LinearKind::Lt)
                    continue;
                auto run = [&](int bank) {
                    auto b = weight(bank);
                    launch_quant_linear(config, {m, n, k}, a.values.data, a.scales.ptr<float>(),
                                        b.values.data, b.scales.ptr<float>(), y.data(),
                                        scratch.data(), scratch.size(), g.stream);
                };
                float us = time_path(g, run, banks);
                if (us < best) {
                    best = us;
                    best_name = path_name(config);
                }
            }
            std::cout << "{\"m\":" << m << ",\"n\":" << n << ",\"k\":" << k
                      << ",\"selected_us\":" << selected << ",\"best_native_us\":" << best
                      << ",\"ratio\":" << selected / best << ",\"best_native\":\"" << best_name
                      << "\"}\n"
                      << std::flush;
        }
    }
}
