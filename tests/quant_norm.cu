#include "backend.h"
#include "quant_reference.h"
#include "graph.h"
#include "quant_norm_config.h"
#include "quant_norm_kernels.cuh"
#include <cassert>
#include <cmath>
#include <iostream>
using namespace tts;
using bf16 = __nv_bfloat16;

template <Type Storage>
void check(Backend &g, int width, bool gemma, int mode, int pattern, bool unaligned) {
    constexpr int rows = 7;
    int offset = unaligned ? 1 : 0;
    Buffer x(size_t(rows) * width * 2 + 2, &g.memory), u(x.size(), &g.memory), w(width * 2 + 2, &g.memory), ls(w.size(), &g.memory),
        q(size_t(rows) * width + 1, &g.memory), s(rows * 4, &g.memory);
    auto xv = x.view(rows, width, Type::BF16, offset * 2),
         uv = u.view(rows, width, Type::BF16, offset * 2),
         wv = w.view(1, width, Type::BF16, offset * 2),
         lv = ls.view(1, width, Type::BF16, offset * 2);
    RowScaledView out{q.view(rows, width, Storage, offset), s.view(rows, 1, Type::F32)};
    std::vector<bf16> hx(rows * width), hu(hx.size()), hw(width), hl(width), expected_x(hx.size()),
        actual_x(hx.size());
    std::vector<float> expected(hx.size()), scales(rows), reference_scales(rows);
    std::vector<unsigned char> data(hx.size());
    for (size_t i = 0; i < hx.size(); ++i) {
        float a = std::sin(i * .037f) * 3.f, b = std::cos(i * .013f) * .1f;
        if (pattern == 1)
            a = b = 0;
        if (pattern == 2) {
            a *= 1e-38f;
            b *= 1e-38f;
        }
        if (pattern == 3) {
            a *= 1e16f;
            b *= 1e16f;
        }
        if (pattern == 4) {
            a = 1.f + (i % 7) * 1e-3f;
            b = -1.f;
        }
        hx[i] = bf16(a);
        hu[i] = bf16(b);
    }
    for (int c = 0; c < width; ++c) {
        float scale = pattern == 5 ? 1e30f : pattern == 6 ? 1e-30f : 1.f;
        hw[c] = bf16(std::cos(c * .029f) * scale);
        hl[c] = bf16(.7f + .2f * std::sin(c * .019f));
    }
    for (int r = 0; r < rows; ++r) {
        double sum = 0;
        for (int c = 0; c < width; ++c) {
            size_t i = size_t(r) * width + c;
            float v = float(hx[i]);
            if (mode) {
                float update = float(hu[i]);
                if (mode == 2)
                    update = float(bf16(update * float(hl[c])));
                v = float(bf16(v + update));
            }
            expected_x[i] = bf16(v);
            sum += double(v) * v;
        }
        float inv = 1.f / std::sqrt(float(sum / width) + 1e-6f), amax = 0;
        for (int c = 0; c < width; ++c) {
            size_t i = size_t(r) * width + c;
            float z = float(expected_x[i]) * inv;
            if (!gemma)
                z = float(bf16(z));
            expected[i] = float(bf16(z * (float(hw[c]) + (gemma ? 1.f : 0.f))));
            amax = std::max(amax, std::abs(expected[i]));
        }
        reference_scales[r] = amax == 0 ? 1.f : std::max(amax / quant_test::range(), FLT_MIN);
    }
    g.upload(uv.data, hu.data(), uv.size());
    g.upload(wv.data, hw.data(), wv.size());
    g.upload(lv.data, hl.data(), lv.size());
    auto verify = [&] {
        g.download(data.data(), out.values.data, out.values.size());
        g.download(scales.data(), s.data(), s.size());
        g.download(actual_x.data(), xv.data, xv.size());
        for (int r = 0; r < rows; ++r) {
            assert(std::isfinite(scales[r]));
            assert(std::abs(scales[r] - reference_scales[r]) <= reference_scales[r] * .001f);
            double err = 0, norm = 0, qerr = 0;
            for (int c = 0; c < width; ++c) {
                size_t i = size_t(r) * width + c;
                assert(float(actual_x[i]) == float(expected_x[i]));
                double ref = expected[i], actual = double(quant_test::decode(data[i])) * scales[r];
                double quant_ref = double(quant_test::decode(quant_test::encode(
                                       expected[i] * (1.f / reference_scales[r])))) *
                                   reference_scales[r];
                err += (actual - ref) * (actual - ref);
                norm += ref * ref;
                qerr += (actual - quant_ref) * (actual - quant_ref);
            }
            assert(err <= .0016 * norm + 1e-90); // Original dequantization tolerance: 4% L2.
            if (qerr > .0001 * norm + 1e-90)
                std::cerr << "quant reference mismatch width=" << width << " gemma=" << gemma
                          << " mode=" << mode << " pattern=" << pattern
                          << " unaligned=" << unaligned << " row=" << r
                          << " relative=" << std::sqrt(qerr / norm) << " scale=" << scales[r]
                          << " expected=" << reference_scales[r] << '\n';
            assert(qerr <= .0001 * norm + 1e-90); // Reduction-order tolerance: 1% L2.
        }
    };
    auto reset = [&] { g.upload(xv.data, hx.data(), xv.size()); };
    if (!unaligned) {
#define RUN(T, V, ADD)                                                                             \
    norm_kernels::residual_rms_quantize<T, V, ADD, Storage><<<rows, T, 0, g.stream>>>(             \
        xv.ptr<bf16>(), mode ? uv.ptr<bf16>() : nullptr, wv.ptr<bf16>(),                           \
        mode == 2 ? lv.ptr<bf16>() : nullptr, out.values.ptr<unsigned char>(), s.ptr<float>(),     \
        width, 1e-6f, gemma)
#define TRY(T, V)                                                                                  \
    if (width % V == 0 && T * V <= 4096) {                                                         \
        reset();                                                                                   \
        if (mode) {                                                                                \
            RUN(T, V, true);                                                                       \
        } else {                                                                                   \
            RUN(T, V, false);                                                                      \
        }                                                                                          \
        verify();                                                                                  \
    }
#define BLOCK(T)                                                                                   \
    TRY(T, 2);                                                                                     \
    TRY(T, 4);                                                                                     \
    TRY(T, 8)
        BLOCK(128);
        BLOCK(256);
        BLOCK(512);
        BLOCK(1024);
#undef BLOCK
#undef TRY
#undef RUN
    }
    reset();
    g.quantized().norm(xv, mode ? uv : TensorView{}, wv, out, 1e-6f, gemma,
                       mode == 2 ? lv : TensorView{});
    verify();
}
int main(int argc, char **argv) {
    Backend g(0, quant_test::quantization(argc, argv));
    for (int width : {31, 128, 1024, 1152, 2048})
        for (bool gemma : {false, true}) {
            g.begin_admission();
            g.quantized().prepare_norm(7, width, gemma);
            for (int mode = 0; mode < 3; ++mode)
                for (int pattern = 0; pattern < 7; ++pattern)
                    for (bool unaligned : {false, true})
                        if (quant_test::storage == Type::I8)
                            check<Type::I8>(g, width, gemma, mode, pattern, unaligned);
                        else
                            check<Type::FP8>(g, width, gemma, mode, pattern, unaligned);
            std::cout << "RMS width=" << width << " gemma=" << gemma
                      << " all modes, distributions, configs and alignment cases passed\n"
                      << std::flush;
        }
    CUDA(cudaGetLastError());
}
