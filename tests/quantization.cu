#include "backends/cuda/backend.h"
#include "quant_reference.h"
#include "kernels/quant/quant_kernel_catalog.h"
#include "kernels/quant/quant_dispatch.h"
#include "kernels/quant/quant_format.cuh"
#include "runtime/graph.h"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
using namespace tts;
using bf16 = __nv_bfloat16;
__global__ void encode_int8_pairs(const float2 *x, unsigned short *q, int pairs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < pairs)
        q[i] = quant_device::Format<Type::I8>::pair(x[i]);
}
void int8_pairs_test(Backend &g) {
    if (g.quantized().type() != Type::I8)
        return;
    std::vector<float> input(65536 + 3 * 512);
    for (unsigned i = 0; i < 65536; ++i) {
        uint32_t bits = i << 16;
        std::memcpy(&input[i], &bits, sizeof(bits));
    }
    for (int i = 0; i < 3 * 512; ++i) {
        float tie = i / 3 - 256.f + .5f;
        input[65536 + i] =
            i % 3 == 0 ? tie : std::nextafter(tie, i % 3 == 1 ? -INFINITY : INFINITY);
    }
    Buffer x(input.size() * sizeof(float), &g.memory), q(input.size(), &g.memory);
    g.upload(x.data(), input.data(), x.size());
    int pairs = int(input.size() / 2);
    encode_int8_pairs<<<(pairs + 255) / 256, 256, 0, g.stream>>>(x.ptr<float2>(),
                                                                 q.ptr<unsigned short>(), pairs);
    std::vector<unsigned char> actual(q.size());
    g.download(actual.data(), q.data(), q.size());
    for (size_t i = 0; i < input.size(); ++i) {
        // CUDA fmaxf suppresses both quiet and signaling NaNs. Host fmaxf may
        // propagate signaling NaNs, so define the expected clipping explicitly.
        float bounded = std::isnan(input[i]) ? -127.f : std::clamp(input[i], -127.f, 127.f);
        auto expected = static_cast<unsigned char>(int(std::nearbyint(bounded)));
        assert(actual[i] == expected);
    }
    std::cout << "INT8 packed conversion: all BF16 patterns and float tie neighbors match "
                 "nearest-even symmetric clipping\n";
}
void linear_test(Backend &g, int rows, int n, int k, bool exact = false, bool projection = false) {
    g.begin_admission();
    Buffer x(size_t(rows) * k * 2, &g.memory), w(size_t(n) * k * 2, &g.memory), qx(size_t(rows) * k, &g.memory), qw(size_t(n) * k, &g.memory),
        sx(rows * 4, &g.memory), sw(n * 4, &g.memory), out(size_t(rows) * n * 2, &g.memory);
    std::vector<bf16> hx(rows * k), hw(size_t(n) * k), actual(rows * n);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < k; ++c)
            hx[r * k + c] = bf16(r || rows == 1 ? std::sin(c * .137 + r) * (r + 1) : 0);
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < k; ++c)
            hw[r * k + c] = bf16(r ? std::cos(c * .07 + r) * (.1 + (r % 97) * .003) : 0);
    g.upload(x.data(), hx.data(), x.size());
    g.upload(w.data(), hw.data(), w.size());
    RowScaledView a{qx.view(rows, k, quant_test::storage), sx.view(rows, 1, Type::F32)},
        b{qw.view(n, k, quant_test::storage), sw.view(n, 1, Type::F32)};
    auto &quant = g.quantized();
    quant.quantize(x.view(rows, k), a);
    quant.quantize(w.view(n, k), b);
    std::vector<unsigned char> ax(rows * k), bw(size_t(n) * k);
    std::vector<float> as(rows), bs(n);
    g.download(ax.data(), qx.data(), qx.size());
    g.download(bw.data(), qw.data(), qw.size());
    g.download(as.data(), sx.data(), sx.size());
    g.download(bs.data(), sw.data(), sw.size());
    if (rows > 1)
        assert(as[0] == 1);
    assert(bs[0] == 1);
    // Host conversion is independent of the GPU reduction and addressing.
    auto check_quant = [&](const std::vector<bf16> &src, const std::vector<unsigned char> &q,
                           const std::vector<float> &scales, int count) {
        for (int r = 0; r < count; ++r) {
            float amax = 0;
            for (int j = 0; j < k; ++j)
                amax = std::max(amax, std::abs(float(src[size_t(r) * k + j])));
            float scale = amax == 0 ? 1 : amax / quant_test::range();
            assert(std::abs(scales[r] - scale) <= scale * 1e-6f);
            float inv = 1.f / scale;
            for (int j = 0; j < k; ++j)
                assert(q[size_t(r) * k + j] ==
                       quant_test::encode(float(src[size_t(r) * k + j]) * inv));
        }
    };
    check_quant(hx, ax, as, rows);
    check_quant(hw, bw, bs, n);
    if (exact) {
        // Small integer products and power-of-two scales isolate orientation
        // from FP8 Tensor Core's finite accumulation quantization.
        for (size_t i = 0; i < ax.size(); ++i)
            ax[i] = quant_test::encode(float(int(i % 5) - 2));
        for (size_t i = 0; i < bw.size(); ++i)
            bw[i] = quant_test::encode(float(int(i % 7) - 3));
        for (int r = 0; r < rows; ++r)
            as[r] = std::ldexp(1.f, -(r % 6));
        for (int c = 0; c < n; ++c)
            bs[c] = std::ldexp(1.f, -(c % 8));
        g.upload(qx.data(), ax.data(), qx.size());
        g.upload(qw.data(), bw.data(), qw.size());
        g.upload(sx.data(), as.data(), sx.size());
        g.upload(sw.data(), bs.data(), sw.size());
    }
    if (projection) {
        quant.prepare_projection(rows, n, k);
        quant.project(x.view(rows, k), b, a, out.view(rows, n));
        // Verify the same prepared path also runs under capture/replay.
        Graph graph;
        graph.capture(g.stream, [&] { quant.project(x.view(rows, k), b, a, out.view(rows, n)); });
        graph.run(g.stream);
    } else {
        quant.prepare(rows, n, k);
        quant.linear(a, b, out.view(rows, n));
    }
    g.download(actual.data(), out.data(), out.size());
    double error = 0, norm = 0;
    float max_error = 0;
    // Cover endpoints and interior channels of large real model projections.
    for (int r = 0; r < rows; ++r)
        for (int sample = 0; sample < std::min(n, 129); ++sample) {
            int c = n <= 129 ? sample : int(size_t(sample) * (n - 1) / 128);
            double sum = 0, sumabs = 0;
            for (int j = 0; j < k; ++j) {
                double p = double(quant_test::decode(ax[r * k + j])) *
                           quant_test::decode(bw[size_t(c) * k + j]);
                sum += p;
                sumabs += std::abs(p);
            }
            float expected = float(bf16(float(sum) * as[r] * bs[c]));
            float diff = float(actual[r * n + c]) - expected;
            assert(std::isfinite(float(actual[r * n + c])));
            // One BF16 ulp plus a bound scaled by absolute dot magnitude handles
            // cancellation. Also enforce a matrix-wide relative error below 0.4%.
            double bound =
                exact ? 0 : std::abs(expected) / 128 + sumabs * as[r] * bs[c] / 4096 + 1e-6;
            if (std::abs(diff) > bound) {
                std::cerr << "GEMM mismatch " << rows << ',' << n << ',' << k << " row=" << r
                          << " col=" << c << " expected=" << expected
                          << " actual=" << float(actual[r * n + c]) << " bound=" << bound << '\n';
                std::abort();
            }
            error += diff * diff;
            norm += expected * expected;
            max_error = std::max(max_error, std::abs(diff));
        }
    double relative = std::sqrt(error / std::max(norm, 1e-30));
    assert(relative < .004);
    std::cout << quantized_name(quant_test::storage) << " linear " << rows << ',' << n << ',' << k
              << " exact=" << exact << " projection=" << projection << " relative L2 " << relative
              << " max " << max_error << '\n';
}
void producer_test(Backend &g, int width, bool gemma) {
    constexpr int rows = 3;
    Buffer x(rows * width * 2, &g.memory), u(x.size(), &g.memory), w(width * 2, &g.memory), reference(x.size(), &g.memory), q(rows * width, &g.memory),
        s(rows * 4, &g.memory), gu(rows * width * 4, &g.memory);
    std::vector<bf16> hx(rows * width), hu(hx.size()), hw(width), actual(rows * width),
        gated(rows * width * 2);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < width; ++c) {
            hx[r * width + c] = bf16(r ? std::sin(c * .041 + r) * r : 0);
            hu[r * width + c] = bf16(std::cos(c * .013 + r) * .1);
        }
    for (int c = 0; c < width; ++c)
        hw[c] = bf16(.7 + .1 * std::sin(c * .029));
    g.upload(x.data(), hx.data(), x.size());
    g.upload(u.data(), hu.data(), u.size());
    g.upload(w.data(), hw.data(), w.size());
    RowScaledView output{q.view(rows, width, quant_test::storage), s.view(rows, 1, Type::F32)};
    auto verify = [&](const char *name) {
        std::vector<unsigned char> data(q.size());
        std::vector<float> scales(rows);
        g.download(data.data(), q.data(), q.size());
        g.download(scales.data(), s.data(), s.size());
        double error = 0, norm = 0;
        for (int r = 0; r < rows; ++r) {
            float amax = 0;
            for (int c = 0; c < width; ++c)
                amax = std::max(amax, std::abs(float(actual[r * width + c])));
            float expected = amax == 0 ? 1 : amax / quant_test::range();
            assert(std::abs(scales[r] - expected) <= expected * .001f);
            for (int c = 0; c < width; ++c) {
                float a = float(actual[r * width + c]);
                float diff = quant_test::decode(data[r * width + c]) * scales[r] - a;
                error += diff * diff;
                norm += a * a;
            }
        }
        double relative = std::sqrt(error / std::max(norm, 1e-30));
        assert(relative < .04);
        std::cout << name << " width=" << width << " gemma=" << gemma << " dequantized relative L2 "
                  << relative << '\n';
    };
    if (width <= 2048) {
        g.residual_norm(x.view(rows, width), u.view(rows, width), w.view(1, width),
                        reference.view(rows, width), 1e-6f, {}, gemma);
        g.download(actual.data(), reference.data(), reference.size());
        g.upload(x.data(), hx.data(), x.size());
        g.quantized().norm(x.view(rows, width), u.view(rows, width), w.view(1, width), output,
                           1e-6f, gemma);
        verify("fused RMS");
    }
    for (size_t i = 0; i < gated.size(); ++i)
        gated[i] = bf16(std::sin(i * .047) * 2);
    g.upload(gu.data(), gated.data(), gu.size());
    g.gated(gu.view(rows, width * 2), reference.view(rows, width), gemma ? 1 : 0);
    g.download(actual.data(), reference.data(), reference.size());
    g.quantized().gated(gu.view(rows, width * 2), output, gemma ? 1 : 0);
    verify("fused gated activation");
}
void graph_after_plan_eviction(Backend &g) {
    constexpr int m = 32, n = 96, k = 128;
    Buffer x(size_t(m) * k, &g.memory), w(size_t(n) * k, &g.memory), sx(m * 4, &g.memory), sw(n * 4, &g.memory), y(size_t(m) * n * 2, &g.memory);
    std::vector<unsigned char> hx(x.size(), quant_test::encode(1.f)),
        hw(w.size(), quant_test::encode(2.f));
    std::vector<float> scales(n, 1.f);
    g.upload(x.data(), hx.data(), x.size());
    g.upload(w.data(), hw.data(), w.size());
    g.upload(sx.data(), scales.data(), sx.size());
    g.upload(sw.data(), scales.data(), sw.size());
    auto &q = g.quantized();
    g.begin_admission();
    q.prepare(m, n, k);
    RowScaledView a{x.view(m, k, quant_test::storage), sx.view(m, 1, Type::F32)},
        b{w.view(n, k, quant_test::storage), sw.view(n, 1, Type::F32)};
    Graph graph;
    graph.capture(g.stream, [&] { q.linear(a, b, y.view(m, n)); });
    graph.run(g.stream);
    g.sync();
    for (int rows = 64; rows < 144; ++rows) {
        g.begin_admission();
        q.prepare(rows, n, k);
    }
    assert(q.cache_stats().prepared_plans <= 65);
    auto searches = q.cache_stats().selection.searches;
    g.begin_admission();
    q.prepare(m, n, k);
    assert(q.cache_stats().selection.searches == searches);
    graph.run(g.stream);
    std::vector<bf16> result(m * n);
    g.download(result.data(), y.data(), y.size());
    for (auto v : result)
        assert(float(v) == 256.f);
    std::cout << "captured quantized graph and algorithm choice survive exact-plan eviction\n";
}
void int8_paths_test(Backend &g) {
    if (g.quantized().type() != Type::I8)
        return;
    cudaDeviceProp device{};
    CUDA(cudaGetDeviceProperties(&device, g.device));
    constexpr int n = 256, k = 8192;
    for (int m : {1, 3, 37, 129}) {
        Buffer x(size_t(m) * k, &g.memory), w(size_t(n) * k, &g.memory), sx(m * 4, &g.memory), sw(n * 4, &g.memory), y(size_t(m) * n * 2, &g.memory),
            workspace(8 << 20, &g.memory);
        std::vector<int8_t> hx(x.size()), hw(w.size());
        std::vector<float> as(m), bs(n);
        for (int r = 0; r < m; ++r) {
            as[r] = std::ldexp(1.f, -(r % 6));
            for (int j = 0; j < k; ++j)
                hx[r * k + j] = r == 0 ? 127 : r == 1 ? -127 : ((j * 47 + r * 31) % 255 - 127);
        }
        for (int c = 0; c < n; ++c) {
            bs[c] = std::ldexp(1.f, -(c % 8));
            for (int j = 0; j < k; ++j)
                hw[c * k + j] = c == 0 ? 127 : c == 1 ? -127 : ((j * 19 + c * 37) % 255 - 127);
        }
        std::vector<bf16> expected(m * n), actual(m * n);
        for (int r = 0; r < m; ++r)
            for (int c = 0; c < n; ++c) {
                int64_t acc = 0;
                for (int j = 0; j < k; ++j)
                    acc += int(hx[r * k + j]) * int(hw[c * k + j]);
                expected[r * n + c] = bf16(float(acc) * as[r] * bs[c]);
            }
        g.upload(x.data(), hx.data(), x.size());
        g.upload(w.data(), hw.data(), w.size());
        g.upload(sx.data(), as.data(), sx.size());
        g.upload(sw.data(), bs.data(), sw.size());
        for (auto path :
             quant_linear_candidates({m, n, k},
                                     {device.major == 9 && device.minor == 0,
                                      size_t(device.sharedMemPerBlock), workspace.size()},
                                     Type::I8)) {
            launch_quant_linear(path, {m, n, k}, x.data(), sx.ptr<float>(), w.data(),
                                sw.ptr<float>(), y.data(), workspace.data(), workspace.size(),
                                g.stream, Type::I8);
            g.download(actual.data(), y.data(), y.size());
            for (size_t i = 0; i < actual.size(); ++i)
                assert(float(actual[i]) == float(expected[i]));
        }
    }
    std::cout
        << "INT8 all GEMV/SM80/Hopper/split-K candidates match INT64 CPU dot, including extrema\n";
}
void int8_rounding_test(Backend &g) {
    if (g.quantized().type() != Type::I8)
        return;
    constexpr int rows = 4, cols = 128, stride = 136;
    Buffer x(rows * stride * 2, &g.memory), q(rows * cols, &g.memory), s(rows * 4, &g.memory);
    std::vector<bf16> host(rows * stride, bf16(0));
    for (int c = 0; c < cols; ++c) {
        // Scale=1 gives exact positive and negative half-integer tie cases.
        host[c] = bf16(c == 0 ? 127.f : (c - 64) * .5f);
        host[stride + c] = bf16(c == 0 ? -127.f : (c - 64) * .5f);
        host[3 * stride + c] = bf16(c & 1 ? 1e-38f : -1e-38f);
    }
    g.upload(x.data(), host.data(), x.size());
    RowScaledView out{q.view(rows, cols, Type::I8), s.view(rows, 1, Type::F32)};
    g.quantized().quantize({x.data(), rows, cols, Type::BF16, stride}, out);
    std::vector<unsigned char> data(q.size());
    std::vector<float> scales(rows);
    g.download(data.data(), q.data(), q.size());
    g.download(scales.data(), s.data(), s.size());
    assert(scales[0] == 1 && scales[1] == 1 && scales[2] == 1);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            assert(data[r * cols + c] ==
                   quant_test::encode(float(host[r * stride + c]) / scales[r]));
    auto wrong = out;
    wrong.values.type = Type::FP8;
    bool rejected = false;
    try {
        g.quantized().quantize(x.view(rows, cols), wrong);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);
    std::cout
        << "INT8 nearest-even ties, zero/subnormal rows, strided source and type rejection pass\n";
}
int main(int argc, char **argv) {
    Backend g(0, quant_test::quantization(argc, argv));
    for (int rows : {1, 3, 7, 32}) {
        linear_test(g, rows, 96, 128);
        linear_test(g, rows, 96, 128, true);
    }
    for (int rows : {1, 3, 8})
        for (auto [n, k] : {std::pair{1536, 1024}, std::pair{1024, 1024}, std::pair{16384, 1024},
                            std::pair{1024, 8192}, std::pair{4096, 2048}, std::pair{2048, 2048},
                            std::pair{12288, 2048}, std::pair{2048, 6144}, std::pair{13824, 1152}})
            linear_test(g, rows, n, k);
    for (int rows : {17, 32}) {
        linear_test(g, rows, 1024, 8192);
        linear_test(g, rows, 16384, 1024);
    }
    // Prefill uses native candidates too. Cover partial token tiles and the
    // non-power-of-two text widths with an independent quantized dot oracle.
    for (int rows : {33, 64, 129, 256})
        for (auto [n, k] : {std::pair{1536, 1152}, std::pair{1152, 1024}, std::pair{13824, 1152},
                            std::pair{1152, 6912}})
            linear_test(g, rows, n, k);
    for (int rows : {1, 3, 7, 17, 32, 64})
        for (auto [n, k] : {std::pair{1024, 1024}, std::pair{2048, 2048}})
            linear_test(g, rows, n, k, false, true);
    for (int width : {1024, 1152, 2048, 6144, 6912, 8192})
        for (bool gemma : {false, true})
            producer_test(g, width, gemma);
    graph_after_plan_eviction(g);
    int8_paths_test(g);
    int8_rounding_test(g);
    int8_pairs_test(g);
    CUDA(cudaGetLastError());
}
