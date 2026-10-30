#pragma once
#include <vector>
namespace tts {
struct GatedConfig {
    int threads = 256, vector = 1, clusters = 1;
    bool table = false;
};
inline std::vector<GatedConfig> gated_candidates(int width, bool cluster_supported) {
    std::vector<GatedConfig> out{{}};
    for (int threads : {128, 256, 512, 1024})
        for (int vector : {2, 4, 8})
            if (width % vector == 0)
                for (bool table : {false, true})
                    out.push_back({threads, vector, 1, table});
    if (cluster_supported)
        for (int clusters : {2, 4, 8})
            for (int threads : {128, 256})
                for (int vector : {4, 8})
                    if (width % vector == 0 && threads * vector * clusters <= 8192)
                        for (bool table : {false, true})
                            out.push_back({threads, vector, clusters, table});
    return out;
}
} // namespace tts
