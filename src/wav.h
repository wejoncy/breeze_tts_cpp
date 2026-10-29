#pragma once
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>
namespace tts {
inline void write_wav(const std::string &path, const std::vector<int16_t> &samples) {
    if (samples.size() > (UINT32_MAX - 36) / 2)
        throw std::runtime_error("WAV exceeds RIFF limit");
    std::ofstream out(std::filesystem::u8path(path), std::ios::binary);
    if (!out)
        throw std::runtime_error("cannot write " + path);
    auto u16 = [&](uint16_t v) {
        char b[] = {char(v), char(v >> 8)};
        out.write(b, 2);
    };
    auto u32 = [&](uint32_t v) {
        char b[] = {char(v), char(v >> 8), char(v >> 16), char(v >> 24)};
        out.write(b, 4);
    };
    out.write("RIFF", 4);
    u32(36 + samples.size() * 2);
    out.write("WAVEfmt ", 8);
    u32(16);
    u16(1);
    u16(1);
    u32(24000);
    u32(48000);
    u16(2);
    u16(16);
    out.write("data", 4);
    u32(samples.size() * 2);
    out.write(reinterpret_cast<const char *>(samples.data()), samples.size() * 2);
    if (!out)
        throw std::runtime_error("failed to write WAV");
}
} // namespace tts
