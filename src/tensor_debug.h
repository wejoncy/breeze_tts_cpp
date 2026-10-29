#pragma once
#include "backend.h"
#include <fstream>
#include <filesystem>
namespace tts {
inline void dump_tensor(Backend &gpu, TensorView x, const std::string &suffix) {
    const char *base = std::getenv("TTS_DUMP_PREFIX");
    if (!base)
        return;
    if (!x.dense())
        throw std::runtime_error("debug dump requires dense tensor");
    std::vector<char> host(x.size());
    gpu.download(host.data(), x.data, x.size());
    std::ofstream out(std::filesystem::u8path(std::string(base) + suffix), std::ios::binary);
    if (!out)
        throw std::runtime_error("cannot create debug dump");
    out.write(host.data(), host.size());
}
} // namespace tts
