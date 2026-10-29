#include "kernel_selection.h"
#include "prepared_cache.h"
#include "quant_kernel_catalog.h"
#include <cassert>
#include <iostream>
#include <set>
using namespace tts;
using Selector = KernelSelector<KernelChoice, KernelOperation>;
int main() {
    KernelSelector<int, int> independent;
    assert(independent.select(7, {3, 16, 128}, [](KernelProblem) { return 42; }) == 42);
    Selector selector;
    std::set<int> measured;
    for (int rows = 1; rows <= 10000; ++rows) {
        auto result = selector.select(KernelOperation::ScaledLinear, {rows, 1024, 8192},
                                      [&](KernelProblem problem) {
                                          assert(problem.rows >= rows);
                                          measured.insert(problem.rows);
                                          return KernelChoice{{LinearKind::Hopper, 16, 64, 1}, 0};
                                      });
        assert(result.linear.channels == 64);
    }
    assert(measured.size() <= 100);
    assert(selector.stats().searches == measured.size());
    assert(selector.stats().entries <= 64);
    assert(selector.stats().hits + selector.stats().searches == 10000);
    auto scale_stats = selector.stats();
    // Workload and operation identity must not alias merely because M matches.
    auto other = selector.select(KernelOperation::Bf16InputProjection, {3, 1024, 8192},
                                 [](KernelProblem) { return KernelChoice{{}, 2}; });
    assert(other.projection_tokens == 2);
    Selector small(2);
    for (int n : {256, 512, 768})
        small.select(KernelOperation::ScaledLinear, {1, n, 128},
                     [](KernelProblem) { return KernelChoice{}; });
    assert(small.stats().entries == 2 && small.stats().evictions == 1);
    try {
        small.select(
            KernelOperation::ScaledLinear, {1, 1024, 128},
            [](KernelProblem) -> KernelChoice { throw std::runtime_error("failed calibration"); });
        assert(false);
    } catch (const std::runtime_error &) {
    }
    assert(small.stats().entries == 2 && small.stats().evictions == 1);
    // Eligibility for the class upper bound must fit every actual member's scratch.
    for (int rows : {3, 7, 17, 31, 33, 65, 129, 257}) {
        auto p = Selector::representative({rows, 1152, 6912});
        auto candidates = quant_linear_candidates(p, {true, 49152, 8u << 20});
        for (const auto &c : candidates)
            if (c.kind == LinearKind::Hopper && c.splits > 1) {
                assert(6912 % (c.splits * 16) == 0);
                assert(size_t(c.splits) * rows * 1152 * 4 <= (8u << 20));
            }
    }
    // Pre-Hopper INT8 always has a Tensor Core plan, including long K and partial M.
    for (int rows : {1, 3, 17, 65, 129, 10000}) {
        auto p = Selector::representative({rows, 1152, 6912});
        auto paths = quant_linear_candidates(p, {false, 49152, 8u << 20}, Type::I8);
        bool tensor_core = false;
        for (auto path : paths) {
            assert(path.kind != LinearKind::Hopper && path.kind != LinearKind::Lt);
            if (path.kind == LinearKind::Sm80) {
                tensor_core = true;
                if (path.splits > 1) {
                    assert(p.reduction % (path.splits * 16) == 0);
                    assert(size_t(path.splits) * p.rows * p.channels * 4 <= (8u << 20));
                }
            }
        }
        assert(tensor_core);
    }
    // Active resources are pinned even when a request needs more than the idle cap.
    PreparedCache<int, int> resources(3);
    for (int i = 0; i < 20; ++i)
        resources.prepare(i, [=] { return std::make_unique<int>(i); });
    for (int i = 0; i < 20; ++i)
        assert(resources.find(i) && *resources.find(i) == i);
    assert(resources.size() == 20);
    for (int request = 0; request < 10000; ++request) {
        resources.begin_admission();
        resources.prepare(100 + request, [=] { return std::make_unique<int>(request); });
        assert(resources.size() <= 4);
        assert(*resources.find(100 + request) == request);
        assert(!resources.find(99 + request)); // idle entries cannot be executed
    }
    assert(resources.evictions() > 0);
    std::cout << "10000 M values: " << scale_stats.searches << " searches, " << scale_stats.hits
              << " hits, " << scale_stats.entries
              << " retained choices; resource/cache bounds and pinning pass\n";
}
