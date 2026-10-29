#include "quant_gated.h"
#include "quant_gated_dispatch.cuh"
#include "kernel_selection.h"
#include "kernel_timing.h"
#include "prepared_cache.h"
#include <array>
#include <cmath>
#include <cstdlib>
#include "log.h"
#include <tuple>
namespace tts {
using bf16 = __nv_bfloat16;
struct QuantGated::Impl {
    DeviceContext &memory;
    Type format;
    cudaStream_t stream;
    bool cluster_supported = false;
    DeviceArena arena;
    std::array<Buffer, 2> tables;
    KernelSelector<GatedConfig, int> selector{32};
    PreparedCache<std::tuple<int, int, int>, GatedConfig> plans{32};
    Impl(DeviceContext &m, Type f)
        : memory(m), format(f), stream(m.stream), arena(m, 2 * 65536 * 2) {
        cluster_supported = gated_dispatch::supports_clusters(m.device);
        for (auto &t : tables)
            t = Buffer(65536 * 2, &arena);
        gated_kernels::initialize_activation_table<0>
            <<<256, 256, 0, stream>>>(tables[0].ptr<bf16>());
        gated_kernels::initialize_activation_table<1>
            <<<256, 256, 0, stream>>>(tables[1].ptr<bf16>());
        CUDA(cudaGetLastError());
    }
    GatedConfig tune(KernelProblem p, int kind) {
        if (!quant_autotune_enabled())
            return {};
        int rows = p.rows, width = p.channels;
        WorkspacePlan size;
        for (size_t n : {size_t(rows) * width * 4 * 4, size_t(rows) * width, size_t(rows) * 4})
            size.allocate(n);
        DeviceArena storage(memory, size.size);
        Buffer x(size_t(rows) * width * 4 * 4, &storage), q(size_t(rows) * width, &storage),
            sc(rows * 4, &storage);
        std::vector<bf16> hx(x.size() / 2);
        for (size_t i = 0; i < hx.size(); ++i)
            hx[i] = bf16(3.f * std::sin(i * .047f));
        CUDA(cudaMemcpyAsync(x.data(), hx.data(), x.size(), cudaMemcpyHostToDevice, stream));
        std::vector<unsigned char> expected(q.size()), actual(q.size());
        std::vector<float> es(rows), as(rows);
        GatedConfig choice;
        float best = INFINITY;
        for (auto c : gated_candidates(width, cluster_supported)) {
            auto run = [&](int bank) {
                gated_dispatch::launch(format, c, kind,
                                       x.ptr<bf16>() + size_t(bank % 4) * rows * width * 2,
                                       q.ptr<unsigned char>(), sc.ptr<float>(), rows, width,
                                       tables[kind].ptr<bf16>(), stream);
            };
            run(0);
            CUDA(
                cudaMemcpyAsync(actual.data(), q.data(), q.size(), cudaMemcpyDeviceToHost, stream));
            CUDA(cudaMemcpyAsync(as.data(), sc.data(), sc.size(), cudaMemcpyDeviceToHost, stream));
            CUDA(cudaStreamSynchronize(stream));
            if (c.vector == 1) {
                expected = actual;
                es = as;
            } else if (actual != expected || as != es)
                throw std::runtime_error("gated candidate numerical mismatch");
            float us = tuning::time_linear(stream, run, 64);
            if (us < best) {
                best = us;
                choice = c;
            }
            TTS_LOG(TTS_LOG_TRACE, "quant.gated",
                    "candidate rows={} width={} kind={} threads={} vector={} clusters={} table={} "
                    "us={}",
                    rows, width, kind, c.threads, c.vector, c.clusters, c.table, us);
        }
        TTS_LOG(TTS_LOG_DEBUG, "quant.gated",
                "selected rows={} width={} kind={} threads={} vector={} clusters={} table={} us={}",
                rows, width, kind, choice.threads, choice.vector, choice.clusters, choice.table,
                best);
        return choice;
    }
};
QuantGated::QuantGated(DeviceContext &memory, Type format)
    : impl(std::make_unique<Impl>(memory, format)) {}
QuantGated::~QuantGated() = default;
void QuantGated::begin_admission() {
    impl->plans.begin_admission();
}
uint64_t QuantGated::execution_revision() const {
    return impl->selector.stats().evictions + impl->plans.evictions();
}
void QuantGated::prepare(int rows, int width, int kind) {
    if (rows < 1 || rows > INT32_MAX / 8 || width < 1 || width > 8192 || kind < 0 || kind > 1)
        throw std::invalid_argument("quantized gated unsupported shape/kind");
    impl->plans.prepare(std::make_tuple(rows, width, kind), [&] {
        return std::make_unique<GatedConfig>(impl->selector.select(
            kind, {rows, width, 1}, [&](KernelProblem p) { return impl->tune(p, kind); }));
    });
}
void QuantGated::run(TensorView input, RowScaledView out, int kind) {
    out.validate();
    if (out.values.type != impl->format || input.type != Type::BF16 || !input.dense() ||
        input.rows != out.values.rows || input.cols != 2 * out.values.cols || kind < 0 || kind > 1)
        throw std::invalid_argument("quantized gated shape/type mismatch");
    auto *plan = impl->plans.find(std::make_tuple(input.rows, out.values.cols, kind));
    if (!plan)
        throw std::logic_error("quantized gated was not prepared before execution");
    auto c = *plan;
    if (uintptr_t(input.data) % (c.vector * 2) || uintptr_t(out.values.data) % c.vector)
        c = {};
    gated_dispatch::launch(impl->format, c, kind, input.ptr<bf16>(),
                           out.values.ptr<unsigned char>(), out.scales.ptr<float>(), input.rows,
                           out.values.cols, impl->tables[kind].ptr<bf16>(), impl->stream);
}
} // namespace tts
