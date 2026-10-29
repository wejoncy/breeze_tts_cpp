#include "backend.h"
#include "quant_gemm.h"
#include "quant_projection.cuh"
#include <cuda_bf16.h>
#include <cmath>
#include <iostream>
using namespace tts;
using bf16 = __nv_bfloat16;
int main(int argc, char **argv) {
    int rows = argc > 1 ? std::atoi(argv[1]) : 8;
    int n = argc > 2 ? std::atoi(argv[2]) : 1024;
    int k = argc > 3 ? std::atoi(argv[3]) : 8192;
    // Explicit native geometry avoids re-tuning under profiler replay, which
    // can otherwise select a different path from the unprofiled engine.
    bool projection = argc > 4 && std::string(argv[4]) == "projection";
    bool native = argc > 4 && !projection;
    int tile_channels = native ? std::atoi(argv[4]) : 64;
    int tile_tokens = argc > 5 ? std::atoi(argv[5]) : 16;
    int splits = argc > 6 ? std::atoi(argv[6]) : 1;
    Backend gpu(0, Quantization::FP8);
    Buffer x(size_t(rows) * k * 2, &gpu.memory), w(size_t(n) * k * 2, &gpu.memory), qx(size_t(rows) * k, &gpu.memory), qw(size_t(n) * k, &gpu.memory),
        sx(rows * 4, &gpu.memory), sw(n * 4, &gpu.memory), y(size_t(rows) * n * 2, &gpu.memory), scratch(8 << 20, &gpu.memory);
    std::vector<bf16> hx(rows * k), hw(size_t(n) * k);
    for (size_t i = 0; i < hx.size(); ++i)
        hx[i] = bf16(std::sin(i * .037));
    for (size_t i = 0; i < hw.size(); ++i)
        hw[i] = bf16(std::cos(i * .017) / std::sqrt(float(k)));
    gpu.upload(x.data(), hx.data(), x.size());
    gpu.upload(w.data(), hw.data(), w.size());
    RowScaledView a{qx.view(rows, k, Type::FP8), sx.view(rows, 1, Type::F32)},
        b{qw.view(n, k, Type::FP8), sw.view(n, 1, Type::F32)};
    auto &quant = gpu.quantized();
    quant.quantize(x.view(rows, k), a);
    quant.quantize(w.view(n, k), b);
    if (!native && !projection)
        quant.prepare(rows, n, k);
    auto run = [&] {
        if (projection) {
            if (k == 1024)
                fp8_projection_token_tile<1024, 4>
                    <<<dim3((n + 7) / 8, (rows + 3) / 4), 256, 0, gpu.stream>>>(
                        x.ptr<bf16>(), qw.ptr<unsigned char>(), sw.ptr<float>(), y.ptr<bf16>(),
                        rows, n);
            else if (k == 2048)
                fp8_projection_token_tile<2048, 4>
                    <<<dim3((n + 7) / 8, (rows + 3) / 4), 256, 0, gpu.stream>>>(
                        x.ptr<bf16>(), qw.ptr<unsigned char>(), sw.ptr<float>(), y.ptr<bf16>(),
                        rows, n);
            else
                throw std::invalid_argument("projection profile width");
        } else if (native)
            fp8_hopper_gemm(qx.data(), sx.ptr<float>(), qw.data(), sw.ptr<float>(), y.data(), rows,
                            n, k, scratch.data(), scratch.size(), gpu.stream, tile_channels,
                            tile_tokens, splits);
        else
            quant.linear(a, b, y.view(rows, n));
    };
    for (int i = 0; i < 5; ++i)
        run();
    gpu.sync();
    {
        Range range("quant.measure");
        run();
        gpu.sync();
    }
    CUDA(cudaGetLastError());
    std::cout << "FP8 kernel shape " << rows << ',' << n << ',' << k << " native=" << native
              << " projection_tokens=" << (projection ? 4 : 0) << " tile=" << tile_channels << 'x'
              << tile_tokens << " splits=" << splits << '\n';
}
