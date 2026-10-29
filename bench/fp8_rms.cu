#include "backend.h"
#include "graph.h"
#include "quant_norm_kernels.cuh"
#include <functional>
#include <iostream>
#include <iomanip>
using namespace tts;
using bf16 = __nv_bfloat16;
__global__ void init(bf16 *x, size_t count, float amplitude, unsigned seed) {
    size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (i < count) {
        unsigned z = unsigned(i) + seed;
        z ^= z >> 16;
        z *= 0x7feb352du;
        z ^= z >> 15;
        z *= 0x846ca68bu;
        z ^= z >> 16;
        x[i] = bf16((float(z & 65535) / 32768.f - 1.f) * amplitude);
    }
}
float timing(Backend &g, const std::function<void()> &run) {
    Graph graph;
    graph.capture(g.stream, [&] {
        for (int j = 0; j < 64; ++j)
            run();
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
        times.push_back(ms * 1000 / 64);
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    std::sort(times.begin(), times.end());
    return times[4];
}
int main(int argc, char **argv) {
    int rows = argc > 1 ? atoi(argv[1]) : 8, width = argc > 2 ? atoi(argv[2]) : 2048;
    bool gemma = argc > 3 ? atoi(argv[3]) : false;
    std::string profile = argc > 4 ? argv[4] : "";
    Backend g(0, Quantization::FP8);
    g.quantized().prepare_norm(rows, width, gemma);
    Buffer x(size_t(rows) * width * 2, &g.memory), u(x.size(), &g.memory), w(width * 2, &g.memory), y(x.size(), &g.memory),
        q(size_t(rows) * width, &g.memory), s(rows * 4, &g.memory);
    auto fill = [&](Buffer &v, float a, unsigned seed) {
        init<<<(v.size() / 2 + 255) / 256, 256, 0, g.stream>>>(v.ptr<bf16>(), v.size() / 2, a,
                                                               seed);
    };
    auto reset = [&] { fill(x, 1, 7); };
    reset();
    fill(u, .1, 11);
    fill(w, .7, 17);
    RowScaledView out{q.view(rows, width, Type::FP8), s.view(rows, 1, Type::F32)};
    std::vector<std::pair<std::string, std::function<void()>>> paths;
    paths.push_back({"bf16", [&] {
                         g.residual_norm(x.view(rows, width), u.view(rows, width), w.view(1, width),
                                         y.view(rows, width), 1e-6f, {}, gemma);
                     }});
    paths.push_back({"current", [&] {
                         g.quantized().norm(x.view(rows, width), u.view(rows, width),
                                            w.view(1, width), out, 1e-6f, gemma);
                     }});
#define CANDIDATE(T, V)                                                                            \
    paths.push_back({"t" #T "v" #V, [&] {                                                          \
                         norm_kernels::residual_rms_quantize<T, V, true>                           \
                             <<<rows, T, 0, g.stream>>>(                                           \
                                 x.ptr<bf16>(), u.ptr<bf16>(), w.ptr<bf16>(), nullptr,             \
                                 q.ptr<unsigned char>(), s.ptr<float>(), width, 1e-6f, gemma);     \
                     }})
    CANDIDATE(64, 2);
    CANDIDATE(64, 4);
    CANDIDATE(64, 8);
    CANDIDATE(128, 2);
    CANDIDATE(128, 4);
    CANDIDATE(128, 8);
    CANDIDATE(256, 2);
    CANDIDATE(256, 4);
    CANDIDATE(256, 8);
    CANDIDATE(512, 2);
    CANDIDATE(512, 4);
    CANDIDATE(1024, 2);
#undef CANDIDATE
    reset();
    paths[1].second();
    std::vector<unsigned char> expected(q.size()), actual(q.size());
    std::vector<float> es(rows), as(rows);
    g.download(expected.data(), q.data(), q.size());
    g.download(es.data(), s.data(), s.size());
    for (auto &[name, run] : paths) {
        if (!profile.empty() && profile != name)
            continue;
        reset();
        run();
        g.sync();
        if (name != "bf16") {
            g.download(actual.data(), q.data(), q.size());
            g.download(as.data(), s.data(), s.size());
            double err = 0, norm = 0;
            for (size_t i = 0; i < q.size(); ++i) {
                __nv_fp8_e4m3 a, b;
                a.__x = actual[i];
                b.__x = expected[i];
                double v = float(a) * as[i / width], ref = float(b) * es[i / width];
                err += (v - ref) * (v - ref);
                norm += ref * ref;
            }
            if (std::sqrt(err / norm) > .01)
                throw std::runtime_error("incorrect RMS candidate " + name);
        }
        if (!profile.empty()) {
            Range range("rms.measure");
            run();
            g.sync();
        } else {
            reset();
            std::cout << std::setprecision(8) << "{\"rows\":" << rows << ",\"width\":" << width
                      << ",\"gemma\":" << gemma << ",\"path\":\"" << name
                      << "\",\"us\":" << timing(g, run) << "}\n"
                      << std::flush;
        }
    }
    CUDA(cudaGetLastError());
}
