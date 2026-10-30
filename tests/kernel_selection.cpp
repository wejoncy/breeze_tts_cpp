#include "kernels/quant/kernel_selection.h"
#include "runtime/graph_shapes.h"
#include "runtime/prepared_cache.h"
#include "kernels/quant/quant_kernel_catalog.h"
#include "kernels/quant/kernel_cost.h"
#include <cassert>
#include <iostream>
#include <set>
using namespace tts;
using Selector = KernelSelector<KernelChoice, KernelOperation>;
int main() {
    // A short K cannot repay another launch; a long K can use idle SMs by
    // partitioning. Verify that the policy responds to hardware and work, not M.
    ChipModel chip{132, 4000000., 14000000., 100000., 1., 7000000., 7000.};
    LinearPath direct{LinearKind::Hopper, 32, 64, 1};
    LinearPath split{LinearKind::Hopper, 32, 64, 2};
    assert(linear_cost({32, 4096, 128}, direct, chip) < linear_cost({32, 4096, 128}, split, chip));
    assert(linear_cost({32, 4096, 16384}, split, chip) <
           linear_cost({32, 4096, 16384}, direct, chip));
    auto warp_mma = direct;
    warp_mma.kind = LinearKind::Sm80;
    assert(linear_cost({32, 4096, 16384}, direct, chip) <
           linear_cost({32, 4096, 16384}, warp_mma, chip));
    auto faster_memory = chip;
    faster_memory.memory_bytes_per_us *= 2;
    assert(linear_cost({32, 4096, 16384}, direct, faster_memory) <
           linear_cost({32, 4096, 16384}, direct, chip));
    assert(select_projection({16, 65536, 1024}, direct, chip) == 0);
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
    assert(selector.stats().entries == measured.size());
    assert(selector.stats().hits + selector.stats().searches == 10000);
    auto scale_stats = selector.stats();
    // Workload and operation identity must not alias merely because M matches.
    auto other = selector.select(KernelOperation::Bf16InputProjection, {3, 1024, 8192},
                                 [](KernelProblem) { return KernelChoice{{}, 2}; });
    assert(other.projection_tokens == 2);
    // Repeatedly interleave model widths and active batch sizes beyond the old
    // global LRU limit. A calibrated class must never search again on turnover.
    Selector turnover;
    for (int pass = 0; pass < 3; ++pass)
        for (int rows = 1; rows <= 16; ++rows)
            for (int n = 256; n <= 4096; n += 256)
                turnover.select(KernelOperation::ScaledLinear, {rows, n, 128}, [&](KernelProblem) {
                    assert(pass == 0);
                    return KernelChoice{};
                });
    assert(turnover.stats().searches == 256 && turnover.stats().hits == 512);
    try {
        turnover.select(
            KernelOperation::ScaledLinear, {1, 8192, 128},
            [](KernelProblem) -> KernelChoice { throw std::runtime_error("failed calibration"); });
        assert(false);
    } catch (const std::runtime_error &) {
    }
    assert(turnover.stats().entries == 256 && turnover.stats().searches == 256);
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
    resources.seal();
    resources.begin_admission();
    assert(*resources.find(10099) == 9999);
    bool created = false, rejected = false;
    try {
        resources.prepare(-1, [&] {
            created = true;
            return std::make_unique<int>(0);
        });
    } catch (const std::logic_error &) {
        rejected = true;
    }
    assert(rejected && !created);
    GraphShapes graph_shapes(10000);
    assert(graph_shapes.rows().size() <= 100);
    for (int rows = 1; rows <= 10000; ++rows) {
        auto codec = graph_shapes.codec(rows);
        assert(codec >= rows && int64_t(codec - rows) * 8 <= rows);
        for (int lanes : {rows, rows + rows / 2, rows * 2, rows * 3}) {
            auto shape = graph_shapes.decoder(rows, lanes, 3);
            if (shape.first) {
                assert(shape.first >= rows && shape.second >= lanes);
                assert(int64_t(shape.first - rows) * 8 <= rows);
                assert(int64_t(shape.second - lanes) * 8 <= lanes);
            }
        }
    }
    GraphShapes small_graphs(8);
    assert(small_graphs.decoder(7, 20, 3) == std::make_pair(7, 21));
    assert(small_graphs.decoder(8, 17, 3) == std::make_pair(0, 0));
    std::cout << "10000 M values: " << scale_stats.searches << " searches, " << scale_stats.hits
              << " hits, " << scale_stats.entries
              << " retained choices; resource/cache bounds and pinning pass\n";
}
