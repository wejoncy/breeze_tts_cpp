#pragma once
#include <cstddef>
namespace tts {
struct KernelProblem {
    int rows, channels, reduction;
};
// Library dispatch cost is an initialization estimate, not a timing result.
// Unknown execution plans retain the conservative two-kernel estimate.
struct LibraryWork {
    double effective_waves = 0;
    unsigned kernel_launches = 2;
};
// Initialized from one compiled kernel instance; independent of request shapes.
struct KernelResources {
    int registers = 0, resident_ctas = 0;
};
struct SelectionStats {
    size_t entries = 0, hits = 0, searches = 0;
};
} // namespace tts
