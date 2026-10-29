// Paired measurements of actual engine operators. CUDA graphs exclude host launch overhead.
#include "backend.h"
#include "graph.h"
#include <cuda_bf16.h>
#include <functional>
#include <iostream>
#include <iomanip>
#include <vector>
using namespace tts;
using bf16 = __nv_bfloat16;
std::string format_name = "fp8";
__global__ void fill_values(bf16 *x, size_t count, unsigned int seed, float scale) {
    size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (i < count) {
        unsigned int z = unsigned(i) + seed;
        z ^= z >> 16;
        z *= 0x7feb352du;
        z ^= z >> 15;
        z *= 0x846ca68bu;
        z ^= z >> 16;
        x[i] = bf16((float(z & 65535) / 32768.f - 1.f) * scale);
    }
}
void fill(Backend &g, Buffer &b, unsigned seed, float scale = 1.f) {
    fill_values<<<(b.size() / 2 + 255) / 256, 256, 0, g.stream>>>(b.ptr<bf16>(), b.size() / 2, seed,
                                                                  scale);
}
float timing(Backend &g, const std::function<void(int)> &run, int count) {
    Graph graph;
    graph.capture(g.stream, [&] {
        for (int j = 0; j < count; ++j)
            run(j);
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
        times.push_back(ms * 1000 / count);
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}
void output(const char *model, const char *op, int m, int n, int k, float b, float q,
            size_t weight_bytes = 0) {
    std::cout << std::setprecision(9) << "{\"model\":\"" << model << "\",\"operator\":\"" << op
              << "\",\"rows\":" << m << ",\"n\":" << n << ",\"k\":" << k << ",\"bf16_us\":" << b
              << ",\"" << format_name << "_us\":" << q << ",\"speedup\":" << (b > 0 ? b / q : 0)
              << ",\"" << format_name
              << "_tflops\":" << (weight_bytes ? 2. * m * n * k / q / 1e6 : 0) << ",\""
              << format_name
              << "_effective_weight_gbps\":" << (weight_bytes ? weight_bytes / q / 1e3 : 0) << "}\n"
              << std::flush;
}
void linear(Backend &g, const char *model, const char *op, int m, int n, int k) {
    cudaDeviceProp property{};
    CUDA(cudaGetDeviceProperties(&property, g.device));
    size_t matrix = size_t(n) * k;
    int banks = std::min<size_t>(
        128, std::max<size_t>(4, (2 * property.l2CacheSize + matrix - 1) / matrix));
    Buffer x(size_t(m) * k * 2, &g.memory), w(matrix * banks * 2, &g.memory), qx(size_t(m) * k, &g.memory), qw(matrix * banks, &g.memory),
        sx(m * 4, &g.memory), sw(size_t(banks) * n * 4, &g.memory), y(size_t(m) * n * 2, &g.memory);
    fill(g, x, 7);
    fill(g, w, 13, .1f);
    auto &quant = g.quantized();
    RowScaledView a{qx.view(m, k, g.quantized().type()), sx.view(m, 1, Type::F32)};
    quant.quantize(x.view(m, k), a);
    quant.quantize(w.view(n * banks, k),
                   {qw.view(n * banks, k, g.quantized().type()), sw.view(n * banks, 1, Type::F32)});
    quant.prepare(m, n, k);
    if (std::string(op) == "out")
        quant.prepare_projection(m, n, k);
    g.sync();
    auto bf = [&](int bank) {
        g.linear(x.view(m, k), w.view(n, k, Type::BF16, matrix * bank * 2), y.view(m, n));
    };
    auto fp = [&](int bank) {
        quant.linear(a,
                     {qw.view(n, k, g.quantized().type(), matrix * bank),
                      sw.view(n, 1, Type::F32, size_t(bank) * n * 4)},
                     y.view(m, n));
    };
    bf(0);
    fp(0);
    g.sync();
    float b = timing(g, bf, banks), q = timing(g, fp, banks);
    output(model, op, m, n, k, b, q, matrix);
    if (std::string(op) == "out") {
        auto complete = [&](int bank) {
            quant.quantize(x.view(m, k), a);
            fp(bank);
        };
        output(model, "out_with_quant", m, n, k, b, timing(g, complete, banks), matrix);
        auto selected = [&](int bank) {
            quant.project(x.view(m, k),
                          {qw.view(n, k, g.quantized().type(), matrix * bank),
                           sw.view(n, 1, Type::F32, size_t(bank) * n * 4)},
                          a, y.view(m, n));
        };
        output(model, "out_complete", m, n, k, b, timing(g, selected, banks), matrix);
    }
}
void producers(Backend &g, const char *model, int rows, int hidden, int intermediate, bool gemma) {
    constexpr int copies = 64, banks = 4;
    auto &quant = g.quantized();
    Buffer x(size_t(banks) * rows * hidden * 2, &g.memory), u(x.size(), &g.memory), w(hidden * 2, &g.memory),
        y(size_t(rows) * hidden * 2, &g.memory), q(size_t(rows) * intermediate, &g.memory), s(rows * 4, &g.memory),
        gu(size_t(banks) * rows * intermediate * 4, &g.memory), gy(size_t(rows) * intermediate * 2, &g.memory);
    fill(g, x, 17);
    fill(g, u, 21, .1);
    fill(g, w, 8, .5);
    fill(g, gu, 43, 2);
    RowScaledView h{q.view(rows, hidden, g.quantized().type()), s.view(rows, 1, Type::F32)},
        z{q.view(rows, intermediate, g.quantized().type()), s.view(rows, 1, Type::F32)};
    auto xv = [&](int i) {
        return x.view(rows, hidden, Type::BF16, size_t(i % banks) * rows * hidden * 2);
    };
    auto uv = [&](int i) {
        return u.view(rows, hidden, Type::BF16, size_t(i % banks) * rows * hidden * 2);
    };
    auto gv = [&](int i) {
        return gu.view(rows, intermediate * 2, Type::BF16,
                       size_t(i % banks) * rows * intermediate * 4);
    };
    quant.prepare_norm(rows, hidden, gemma);
    auto normbf = [&](int i) {
        g.norm(xv(i), w.view(1, hidden), y.view(rows, hidden), 1e-6f, gemma);
    };
    auto normfp = [&](int i) { quant.norm(xv(i), {}, w.view(1, hidden), h, 1e-6f, gemma); };
    float b = timing(g, normbf, copies), f = timing(g, normfp, copies);
    output(model, "rms", rows, hidden, 0, b, f);
    auto resbf = [&](int i) {
        g.residual_norm(xv(i), uv(i), w.view(1, hidden), y.view(rows, hidden), 1e-6f, {}, gemma);
    };
    auto resfp = [&](int i) { quant.norm(xv(i), uv(i), w.view(1, hidden), h, 1e-6f, gemma); };
    fill(g, x, 17);
    b = timing(g, resbf, copies);
    fill(g, x, 17);
    f = timing(g, resfp, copies);
    output(model, "residual_rms", rows, hidden, 0, b, f);
    quant.prepare_gated(rows, intermediate, gemma ? 1 : 0);
    auto gatebf = [&](int i) { g.gated(gv(i), gy.view(rows, intermediate), gemma ? 1 : 0); };
    auto gatefp = [&](int i) { quant.gated(gv(i), z, gemma ? 1 : 0); };
    b = timing(g, gatebf, copies);
    f = timing(g, gatefp, copies);
    output(model, "gated_activation", rows, intermediate, 0, b, f);
    auto convert = [&](int i) { quant.quantize(xv(i), h); };
    output(model, "standalone_dynamic_quant", rows, hidden, 0, 0, timing(g, convert, copies));
}
int main(int argc, char **argv) {
    auto quantization = quantization_from_string(argc > 2 ? argv[2] : "fp8");
    format_name = argc > 2 ? argv[2] : "fp8";
    Backend g(0, quantization);
    bool text_only = argc > 1 && std::string(argv[1]) == "text";
    std::vector<int> batches = text_only  ? std::vector<int>{}
                               : argc > 1 ? std::vector<int>{std::atoi(argv[1])}
                                          : std::vector<int>{1, 4, 8, 32};
    for (int m : batches) {
        linear(g, "depth", "qkv", m, 1536, 1024);
        linear(g, "depth", "out", m, 1024, 1024);
        linear(g, "depth", "gate_up", m, 16384, 1024);
        linear(g, "depth", "down", m, 1024, 8192);
        linear(g, "backbone", "qkv", m, 4096, 2048);
        linear(g, "backbone", "out", m, 2048, 2048);
        linear(g, "backbone", "gate_up", m, 12288, 2048);
        linear(g, "backbone", "down", m, 2048, 6144);
        producers(g, "depth", m, 1024, 8192, false);
        producers(g, "backbone", m, 2048, 6144, false);
    }
    if (argc == 1 || text_only)
        for (int m : {64, 256}) {
            linear(g, "text", "qkv", m, 1536, 1152);
            linear(g, "text", "out", m, 1152, 1024);
            linear(g, "text", "gate_up", m, 13824, 1152);
            linear(g, "text", "down", m, 1152, 6912);
            producers(g, "text", m, 1152, 6912, true);
        }
    CUDA(cudaGetLastError());
}
