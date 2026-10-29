#include "backend.h"
#include "quant_reference.h"
#include "quant_gated_dispatch.cuh"
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
using namespace tts;
using bf16 = __nv_bfloat16;
void table_test(Backend &g, int kind, Buffer &table) {
    constexpr int rows = 64, width = 1024;
    Buffer x(size_t(rows) * width * 4, &g.memory), y(size_t(rows) * width * 2, &g.memory);
    std::vector<bf16> hx(x.size() / 2), actual(rows * width), lookup(rows * width);
    for (int b = 0; b < 65536; ++b) {
        unsigned short bits = b;
        bf16 v;
        std::memcpy(&v, &bits, 2);
        int row = b / width, col = b % width;
        hx[row * width * 2 + col] = ((b & 0x7f80) == 0x7f80) ? bf16(0) : v;
        hx[row * width * 2 + width + col] = bf16(1);
    }
    g.upload(x.data(), hx.data(), x.size());
    g.gated(x.view(rows, width * 2), y.view(rows, width), kind);
    g.download(actual.data(), y.data(), y.size());
    g.download(lookup.data(), table.data(), table.size());
    for (int b = 0; b < 65536; ++b)
        if ((b & 0x7f80) != 0x7f80)
            assert(std::memcmp(&actual[b], &lookup[b], 2) == 0);
    std::cout << "kind=" << kind << " all 65280 finite BF16 inputs match unary table bit-for-bit\n";
}
void check(Backend &g, int width, int kind, int pattern, Buffer &table) {
    constexpr int rows = 3;
    Buffer input(size_t(rows) * width * 4 + 2, &g.memory), reference(size_t(rows) * width * 2, &g.memory),
        expected_q(size_t(rows) * width, &g.memory), expected_s(rows * 4, &g.memory), q(expected_q.size() + 1, &g.memory),
        s(rows * 4, &g.memory);
    std::vector<bf16> hx(size_t(rows) * width * 2);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < width; ++c) {
            float a = 8.f * std::sin((r * width + c) * .037f),
                  b = std::cos((r * width + c) * .029f) * 2;
            if (pattern == 1)
                a = b = 0;
            if (pattern == 2) {
                a *= 1e-38f;
                b = 1;
            }
            if (pattern == 3) {
                unsigned short bits = (r * width + c) * 53;
                bf16 v;
                std::memcpy(&v, &bits, 2);
                a = std::isfinite(float(v)) ? float(v) : 0.f;
                b = 1;
            }
            hx[r * width * 2 + c] = bf16(a);
            hx[r * width * 2 + width + c] = bf16(b);
        }
    g.upload(input.data(), hx.data(), hx.size() * 2);
    g.gated(input.view(rows, width * 2), reference.view(rows, width), kind);
    RowScaledView expected{expected_q.view(rows, width, quant_test::storage),
                           expected_s.view(rows, 1, Type::F32)};
    g.quantized().quantize(reference.view(rows, width), expected);
    std::vector<unsigned char> eq(expected_q.size()), aq(eq.size());
    std::vector<float> es(rows), as(rows);
    g.download(eq.data(), expected_q.data(), expected_q.size());
    g.download(es.data(), expected_s.data(), expected_s.size());
    auto verify = [&](void *values) {
        g.download(aq.data(), values, aq.size());
        g.download(as.data(), s.data(), s.size());
        assert(aq == eq);
        assert(as == es);
    };
    for (auto c : gated_candidates(width, gated_dispatch::supports_clusters(g.device))) {
        gated_dispatch::launch(quant_test::storage, c, kind, input.ptr<bf16>(),
                               q.ptr<unsigned char>(), s.ptr<float>(), rows, width,
                               table.ptr<bf16>(), g.stream);
        verify(q.data());
    }
    for (int offset : {0, 1}) {
        auto x = input.view(rows, width * 2, Type::BF16, offset * 2);
        RowScaledView output{q.view(rows, width, quant_test::storage, offset),
                             s.view(rows, 1, Type::F32)};
        g.upload(x.data, hx.data(), hx.size() * 2);
        g.quantized().gated(x, output, kind);
        verify(output.values.data);
    }
}
int main(int argc, char **argv) {
    Backend g(0, quant_test::quantization(argc, argv));
    Buffer table(65536 * 2, &g.memory);
    for (int kind : {0, 1}) {
        if (kind)
            gated_kernels::initialize_activation_table<1>
                <<<256, 256, 0, g.stream>>>(table.ptr<bf16>());
        else
            gated_kernels::initialize_activation_table<0>
                <<<256, 256, 0, g.stream>>>(table.ptr<bf16>());
        table_test(g, kind, table);
        for (int width : {31, 128, 6144, 6912, 8192}) {
            g.begin_admission();
            g.quantized().prepare_gated(3, width, kind);
            for (int pattern = 0; pattern < 4; ++pattern)
                check(g, width, kind, pattern, table);
            std::cout << "gated width=" << width << " kind=" << kind
                      << " all candidates/distributions/alignment paths bit-exact\n"
                      << std::flush;
        }
    }
    CUDA(cudaGetLastError());
}
