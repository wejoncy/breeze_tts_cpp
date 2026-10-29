#include "audio_io.h"
#include <filesystem>
#include <sndfile.h>
#include <soxr.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
namespace tts {
namespace {
struct Source {
    const char *data;
    sf_count_t size, pos = 0;
};
sf_count_t length(void *u) {
    return static_cast<Source *>(u)->size;
}
sf_count_t seek(sf_count_t off, int whence, void *u) {
    auto &s = *static_cast<Source *>(u);
    auto p = (whence == SEEK_SET ? 0 : whence == SEEK_CUR ? s.pos : s.size) + off;
    if (p < 0 || p > s.size)
        return -1;
    return s.pos = p;
}
sf_count_t read(void *p, sf_count_t n, void *u) {
    auto &s = *static_cast<Source *>(u);
    n = std::max<sf_count_t>(0, std::min(n, s.size - s.pos));
    memcpy(p, s.data + s.pos, n);
    s.pos += n;
    return n;
}
sf_count_t tell(void *u) {
    return static_cast<Source *>(u)->pos;
}
} // namespace
std::vector<float> resample_audio(const float *input, size_t frames, int rate, int channels,
                                  size_t limit) {
    if (!input || !frames || rate < 1 || rate > 768000 || channels < 1 || channels > 256)
        throw std::invalid_argument("invalid reference PCM");
    double output_frames = std::ceil(double(frames) * 24000 / rate);
    if (output_frames > double(limit))
        throw std::invalid_argument("reference audio exceeds context capacity");
    size_t count = size_t(output_frames);
    std::vector<float> mono(frames);
    for (size_t t = 0; t < frames; t++) {
        double sum = 0;
        for (int c = 0; c < channels; c++) {
            float v = input[t * channels + c];
            if (!std::isfinite(v))
                throw std::invalid_argument("nonfinite reference PCM");
            sum += v;
        }
        mono[t] = sum / channels;
    }
    if (rate == 24000)
        return mono;
    auto quality = soxr_quality_spec(SOXR_HQ, 0);
    std::vector<float> result(count);
    size_t used = 0, written = 0;
    auto error = soxr_oneshot(rate, 24000, 1, mono.data(), frames, &used, result.data(), count,
                              &written, nullptr, &quality, nullptr);
    if (error || used != frames)
        throw std::runtime_error(error ? error : "incomplete audio resampling");
    // librosa fix_length pads the rounded output to ceil(original * ratio).
    std::fill(result.begin() + written, result.end(), 0);
    return result;
}
std::vector<float> read_audio(const void *data, size_t bytes, size_t limit) {
    if (!data || !bytes || bytes > size_t(1) << 30)
        throw std::invalid_argument("invalid encoded reference audio size");
    Source source{static_cast<const char *>(data), sf_count_t(bytes)};
    SF_VIRTUAL_IO io{length, seek, read, nullptr, tell};
    SF_INFO info{};
    std::unique_ptr<SNDFILE, decltype(&sf_close)> file(
        sf_open_virtual(&io, SFM_READ, &info, &source), sf_close);
    if (!file)
        throw std::invalid_argument("cannot decode reference audio");
    if (info.frames < 1 || info.channels < 1 || info.channels > 256 || info.samplerate < 1 ||
        info.samplerate > 768000 || double(info.frames) * 24000 / info.samplerate > limit)
        throw std::invalid_argument("invalid or overlong reference audio");
    std::vector<float> input(size_t(info.frames) * info.channels);
    if (sf_readf_float(file.get(), input.data(), info.frames) != info.frames)
        throw std::invalid_argument("truncated reference audio");
    return resample_audio(input.data(), info.frames, info.samplerate, info.channels, limit);
}
std::vector<float> read_audio_file(const std::string &path, size_t limit) {
    std::ifstream f(std::filesystem::u8path(path), std::ios::binary | std::ios::ate);
    if (!f)
        throw std::runtime_error("cannot open reference audio: " + path);
    auto size = f.tellg();
    if (size <= 0 || size > 1LL << 30)
        throw std::runtime_error("invalid reference file size");
    std::vector<char> data(size);
    f.seekg(0);
    if (!f.read(data.data(), size))
        throw std::runtime_error("cannot read reference audio");
    return read_audio(data.data(), data.size(), limit);
}
} // namespace tts
