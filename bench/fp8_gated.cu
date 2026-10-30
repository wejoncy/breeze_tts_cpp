#include "backends/cuda/backend.h"
#include "runtime/graph.h"
#include "support/quant_gated_dispatch.cuh"
#include "gated_legacy.cuh"
#include "support/kernel_timing.h"
#include <functional>
#include <iostream>
#include <iomanip>
using namespace tts;
using bf16 = __nv_bfloat16;
int main(int argc, char **argv) {
    int rows = argc > 1 ? atoi(argv[1]) : 8, width = argc > 2 ? atoi(argv[2]) : 8192,
        kind = argc > 3 ? atoi(argv[3]) : 0;
    std::string profile = argc > 4 ? argv[4] : "";
    Backend g(0, Quantization::FP8);
    Buffer x(size_t(rows) * width * 4, &g.memory), y(size_t(rows) * width * 2, &g.memory), q(size_t(rows) * width, &g.memory),
        s(rows * 4, &g.memory);
    Buffer table(65536 * 2, &g.memory);
    if (kind)
        gated_kernels::initialize_activation_table<1><<<256, 256, 0, g.stream>>>(table.ptr<bf16>());
    else
        gated_kernels::initialize_activation_table<0><<<256, 256, 0, g.stream>>>(table.ptr<bf16>());
    std::vector<bf16> hx(x.size() / 2);
    for (size_t i = 0; i < hx.size(); ++i)
        hx[i] = bf16(3.f * std::sin(i * .047f));
    g.upload(x.data(), hx.data(), x.size());
    RowScaledView out{q.view(rows, width, Type::FP8), s.view(rows, 1, Type::F32)};
    std::vector<std::pair<std::string, std::function<void()>>> paths;
    paths.push_back({"bf16", [&] { g.gated(x.view(rows, width * 2), y.view(rows, width), kind); }});
    paths.push_back({"current", [&] { g.quantized().gated(x.view(rows, width * 2), out, kind); }});
    if (width % 4 == 0)
        paths.push_back({"legacy", [&] {
                             if (kind)
                                 bench::legacy_gated_quantize<1><<<rows, 1024, 0, g.stream>>>(
                                     x.ptr<bf16>(), q.ptr<unsigned char>(), s.ptr<float>(), width);
                             else
                                 bench::legacy_gated_quantize<0><<<rows, 1024, 0, g.stream>>>(
                                     x.ptr<bf16>(), q.ptr<unsigned char>(), s.ptr<float>(), width);
                         }});
    int cluster_supported = 0;
    CUDA(cudaDeviceGetAttribute(&cluster_supported, cudaDevAttrClusterLaunch, 0));
    for (auto config : gated_candidates(width, cluster_supported)) {
        std::string name =
            "t" + std::to_string(config.threads) + "v" + std::to_string(config.vector);
        if (config.vector == 1)
            name = "scalar";
        else if (config.clusters > 1)
            name = "cluster" + std::to_string(config.clusters) + "_" + name + "lut" +
                   (config.table ? "true" : "false");
        else if (config.table)
            name = "lut_" + name;
        paths.push_back({name, [&, config] {
                             gated_dispatch::launch(config, kind, x.ptr<bf16>(),
                                                    q.ptr<unsigned char>(), s.ptr<float>(), rows,
                                                    width, table.ptr<bf16>(), g.stream);
                         }});
    }
    if (!profile.empty() &&
        std::none_of(paths.begin(), paths.end(), [&](const auto &p) { return p.first == profile; }))
        throw std::invalid_argument("unknown gated profile path: " + profile);
    paths[1].second();
    std::vector<unsigned char> expected(q.size()), actual(q.size());
    std::vector<float> es(rows), as(rows);
    g.download(expected.data(), q.data(), q.size());
    g.download(es.data(), s.data(), s.size());
    for (auto &path : paths) {
        const auto &name = path.first;
        auto run = path.second;
        if (!profile.empty() && profile != name)
            continue;
        run();
        g.sync();
        if (name != "bf16") {
            g.download(actual.data(), q.data(), q.size());
            g.download(as.data(), s.data(), s.size());
            if (actual != expected || as != es)
                throw std::runtime_error("incorrect gated candidate " + name);
        }
        if (!profile.empty()) {
            Range range("gated.measure");
            run();
            g.sync();
        } else
            std::cout << std::setprecision(8) << "{\"rows\":" << rows << ",\"width\":" << width
                      << ",\"kind\":" << kind << ",\"path\":\"" << name << "\",\"us\":"
                      << tuning::time_linear(
                             g.stream, [&](int) { run(); }, 64)
                      << "}\n"
                      << std::flush;
    }
    CUDA(cudaGetLastError());
}
