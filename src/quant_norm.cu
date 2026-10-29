#include "quant_norm.h"
#include "quant_norm_config.h"
#include "quant_norm_kernels.cuh"
#include "kernel_selection.h"
#include "kernel_timing.h"
#include "prepared_cache.h"
#include <array>
#include <cmath>
#include <cstdlib>
#include "log.h"
#include <tuple>

namespace tts {
namespace {
using bf16 = __nv_bfloat16;
struct NormPlan {
    std::array<NormConfig, 3> modes;
};
template <Type Storage>
void launch_format(NormConfig config, bf16 *x, const bf16 *u, const bf16 *w, const bf16 *scale,
                   unsigned char *q, float *s, int rows, int width, float eps, bool gemma,
                   cudaStream_t stream) {
    if (config.vector == 1) {
        norm_kernels::legacy_residual_rms_quantize<Storage>
            <<<rows, 256, 0, stream>>>(x, u, w, scale, q, s, width, eps, gemma);
        return;
    }
#define RUN(T, V)                                                                                  \
    if (u) {                                                                                       \
        norm_kernels::residual_rms_quantize<T, V, true, Storage>                                   \
            <<<rows, T, 0, stream>>>(x, u, w, scale, q, s, width, eps, gemma);                     \
    } else {                                                                                       \
        norm_kernels::residual_rms_quantize<T, V, false, Storage>                                  \
            <<<rows, T, 0, stream>>>(x, u, w, scale, q, s, width, eps, gemma);                     \
    }
#define WIDTH(T)                                                                                   \
    case T:                                                                                        \
        switch (config.vector) {                                                                   \
        case 2:                                                                                    \
            RUN(T, 2);                                                                             \
            break;                                                                                 \
        case 4:                                                                                    \
            RUN(T, 4);                                                                             \
            break;                                                                                 \
        case 8:                                                                                    \
            RUN(T, 8);                                                                             \
            break;                                                                                 \
        default:                                                                                   \
            throw std::logic_error("invalid RMS vector");                                          \
        }                                                                                          \
        break
    switch (config.threads) {
        WIDTH(128);
        WIDTH(256);
        WIDTH(512);
        WIDTH(1024);
    default:
        throw std::logic_error("invalid RMS block");
    }
#undef WIDTH
#undef RUN
}
void launch(Type format, NormConfig config, bf16 *x, const bf16 *u, const bf16 *w,
            const bf16 *scale, unsigned char *q, float *s, int rows, int width, float eps,
            bool gemma, cudaStream_t stream) {
    if (format == Type::I8)
        launch_format<Type::I8>(config, x, u, w, scale, q, s, rows, width, eps, gemma, stream);
    else
        launch_format<Type::FP8>(config, x, u, w, scale, q, s, rows, width, eps, gemma, stream);
}
float quant_value(Type format, unsigned char x) {
    return format == Type::I8 ? quant_device::Format<Type::I8>::decode(x)
                              : quant_device::Format<Type::FP8>::decode(x);
}
} // namespace
struct QuantNorm::Impl {
    DeviceContext &memory;
    Type format;
    cudaStream_t stream;
    KernelSelector<NormConfig, int> selector{32};
    PreparedCache<std::tuple<int, int, bool>, NormPlan> plans{32};
    Impl(DeviceContext &m, Type f) : memory(m), format(f), stream(m.stream) {}
    NormConfig tune(KernelProblem p, int mode, bool gemma) {
        auto candidates = norm_candidates(p.channels);
        if (!quant_autotune_enabled())
            return {};
        int rows = p.rows, width = p.channels;
        constexpr int banks = 4;
        size_t bank_bytes = size_t(rows) * width * 2;
        WorkspacePlan size;
        for (size_t bytes : {bank_bytes * banks, bank_bytes * banks, size_t(width) * 2,
                             size_t(width) * 2, size_t(rows) * width, size_t(rows) * 4})
            size.allocate(bytes);
        DeviceArena storage(memory, size.size);
        Buffer x(bank_bytes * banks, &storage), u(x.size(), &storage), w(width * 2, &storage),
            sc(width * 2, &storage), q(size_t(rows) * width, &storage), s(rows * 4, &storage);
        std::vector<bf16> hx(x.size() / 2), hu(hx.size()), hw(width), hs(width);
        for (size_t i = 0; i < hx.size(); ++i) {
            hx[i] = bf16(std::sin(i * .037));
            hu[i] = bf16(.1f * std::cos(i * .013));
        }
        for (int i = 0; i < width; ++i) {
            hw[i] = bf16(.7f + .1f * std::sin(i * .029));
            hs[i] = bf16(.8f + .1f * std::cos(i * .019));
        }
        auto upload = [&](Buffer &b, const auto &v) {
            CUDA(cudaMemcpyAsync(b.data(), v.data(), b.size(), cudaMemcpyHostToDevice, stream));
        };
        upload(u, hu);
        upload(w, hw);
        upload(sc, hs);
        std::vector<unsigned char> expected(q.size()), actual(q.size());
        std::vector<float> es(rows), as(rows);
        float best = INFINITY;
        NormConfig choice;
        for (auto cfg : candidates) {
            upload(x, hx);
            auto run = [&](int bank) {
                launch(format, cfg, x.ptr<bf16>() + size_t(bank % banks) * rows * width,
                       mode ? u.ptr<bf16>() + size_t(bank % banks) * rows * width : nullptr,
                       w.ptr<bf16>(), mode == 2 ? sc.ptr<bf16>() : nullptr, q.ptr<unsigned char>(),
                       s.ptr<float>(), rows, width, 1e-6f, gemma, stream);
            };
            run(0);
            CUDA(
                cudaMemcpyAsync(actual.data(), q.data(), q.size(), cudaMemcpyDeviceToHost, stream));
            CUDA(cudaMemcpyAsync(as.data(), s.data(), s.size(), cudaMemcpyDeviceToHost, stream));
            CUDA(cudaStreamSynchronize(stream));
            if (cfg.vector == 1) {
                expected = actual;
                es = as;
            } else {
                double error = 0, norm = 0;
                for (int r = 0; r < rows; ++r)
                    if (!std::isfinite(as[r]) || std::abs(as[r] - es[r]) > es[r] * .001f)
                        throw std::runtime_error("RMS candidate scale mismatch");
                for (size_t i = 0; i < q.size(); ++i) {
                    double a = quant_value(format, actual[i]) * as[i / width],
                           b = quant_value(format, expected[i]) * es[i / width];
                    error += (a - b) * (a - b);
                    norm += b * b;
                }
                if (error > .0001 * std::max(norm, 1e-30))
                    throw std::runtime_error("RMS candidate numerical mismatch");
            }
            upload(x, hx);
            float us = tuning::time_linear(stream, run, 64);
            if (us < best) {
                best = us;
                choice = cfg;
            }
            TTS_LOG(TTS_LOG_TRACE, "quant.norm",
                    "candidate rows={} width={} mode={} gemma={} threads={} vector={} us={}", rows,
                    width, mode, gemma, cfg.threads, cfg.vector, us);
        }
        TTS_LOG(TTS_LOG_DEBUG, "quant.norm",
                "selected rows={} width={} mode={} gemma={} threads={} vector={} us={}", rows,
                width, mode, gemma, choice.threads, choice.vector, best);
        return choice;
    }
};
QuantNorm::QuantNorm(DeviceContext &memory, Type format)
    : impl(std::make_unique<Impl>(memory, format)) {}
QuantNorm::~QuantNorm() = default;
void QuantNorm::begin_admission() {
    impl->plans.begin_admission();
}
uint64_t QuantNorm::execution_revision() const {
    return impl->selector.stats().evictions + impl->plans.evictions();
}
void QuantNorm::prepare(int rows, int width, bool gemma) {
    if (rows < 1 || width < 1 || width > 2048)
        throw std::invalid_argument("quantized RMS unsupported shape");
    impl->plans.prepare(std::make_tuple(rows, width, gemma), [&] {
        auto p = std::make_unique<NormPlan>();
        for (int mode = 0; mode < 3; ++mode)
            p->modes[mode] = impl->selector.select(
                mode + 3 * int(gemma), {rows, width, 1},
                [&](KernelProblem shape) { return impl->tune(shape, mode, gemma); });
        return p;
    });
}
void QuantNorm::run(TensorView x, TensorView u, TensorView w, RowScaledView out, float eps,
                    bool gemma, TensorView scale) {
    out.validate();
    if (out.values.type != impl->format || x.type != Type::BF16 || !x.dense() || x.cols > 2048 ||
        x.cols != out.values.cols || x.rows != out.values.rows || w.type != Type::BF16 ||
        !w.dense() || w.numel() != size_t(x.cols) ||
        (u.data && (u.type != x.type || u.rows != x.rows || u.cols != x.cols || !u.dense())) ||
        (scale.data &&
         (scale.type != Type::BF16 || !scale.dense() || scale.numel() != size_t(x.cols))))
        throw std::invalid_argument("quantized RMS shape/type mismatch");
    auto *plan = impl->plans.find(std::make_tuple(x.rows, x.cols, gemma));
    if (!plan)
        throw std::logic_error("quantized RMS was not prepared before execution");
    auto cfg = plan->modes[u.data ? (scale.data ? 2 : 1) : 0];
    for (const void *p : {x.data, u.data, w.data, scale.data})
        if (p && uintptr_t(p) % (cfg.vector * 2))
            cfg = {};
    if (uintptr_t(out.values.data) % cfg.vector)
        cfg = {};
    launch(impl->format, cfg, x.ptr<bf16>(), u.ptr<bf16>(), w.ptr<bf16>(), scale.ptr<bf16>(),
           out.values.ptr<unsigned char>(), out.scales.ptr<float>(), x.rows, x.cols, eps, gemma,
           impl->stream);
}
} // namespace tts
