#pragma once
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
namespace tts {
inline void write_wav(std::ostream &out, std::string_view pcm, int sample_rate) {
    if (sample_rate <= 0 || sample_rate > INT32_MAX / 2)
        throw std::invalid_argument("invalid WAV sample rate");
    if (pcm.size() > UINT32_MAX - 36 || pcm.size() % 2)
        throw std::runtime_error("WAV exceeds RIFF limit");
    auto u16 = [&](uint16_t v) {
        char b[] = {char(v), char(v >> 8)};
        out.write(b, 2);
    };
    auto u32 = [&](uint32_t v) {
        char b[] = {char(v), char(v >> 8), char(v >> 16), char(v >> 24)};
        out.write(b, 4);
    };
    out.write("RIFF", 4);
    u32(36 + pcm.size());
    out.write("WAVEfmt ", 8);
    u32(16);
    u16(1);
    u16(1);
    u32(sample_rate);
    u32(sample_rate * sizeof(int16_t));
    u16(2);
    u16(16);
    out.write("data", 4);
    u32(pcm.size());
    out.write(pcm.data(), pcm.size());
    if (!out)
        throw std::runtime_error("failed to write WAV");
}
inline void write_wav(const std::string &path, const std::vector<int16_t> &samples,
                      int sample_rate) {
    std::ofstream out(std::filesystem::u8path(path), std::ios::binary);
    if (!out)
        throw std::runtime_error("cannot write " + path);
    write_wav(out, {reinterpret_cast<const char *>(samples.data()), samples.size() * 2},
              sample_rate);
}
} // namespace tts
