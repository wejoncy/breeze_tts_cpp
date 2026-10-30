#pragma once
#include "backends/cuda/backend.h"
#include <fstream>
#include <filesystem>
namespace tts {
inline thread_local uint64_t tensor_debug_request = 0;
struct TensorDebugScope {
    uint64_t previous = tensor_debug_request;
    explicit TensorDebugScope(uint64_t request) {
        tensor_debug_request = request;
    }
    ~TensorDebugScope() {
        tensor_debug_request = previous;
    }
};
inline std::string debug_path(std::string path, uint64_t request) {
    if (auto marker = path.find("{request}"); marker != std::string::npos)
        path.replace(marker, 9, std::to_string(request));
    return path;
}
inline void dump_tensor(Backend &gpu, TensorView x, const std::string &suffix) {
    const char *base = std::getenv("TTS_DUMP_PREFIX");
    if (!base)
        return;
    if (!x.dense())
        throw std::runtime_error("debug dump requires dense tensor");
    std::vector<char> host(x.size());
    gpu.download(host.data(), x.data, x.size());
    std::ofstream out(std::filesystem::u8path(debug_path(base, tensor_debug_request) + suffix),
                      std::ios::binary);
    if (!out)
        throw std::runtime_error("cannot create debug dump");
    out.write(host.data(), host.size());
}
} // namespace tts
