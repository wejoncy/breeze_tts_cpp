#pragma once
#include <vector>
namespace tts {
struct NormConfig {
    int threads = 256, vector = 1;
};
inline std::vector<NormConfig> norm_candidates(int width) {
    std::vector<NormConfig> out{{256, 1}};
    for (int threads : {128, 256, 512, 1024})
        for (int vector : {2, 4, 8})
            if (width % vector == 0 && threads * vector <= 4096)
                out.push_back({threads, vector});
    return out;
}
} // namespace tts
