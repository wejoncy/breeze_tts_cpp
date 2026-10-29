#pragma once
#include <cstddef>
#include <string>
#include <vector>
namespace tts {
std::vector<float> resample_audio(const float *interleaved, size_t frames, int rate, int channels,
                                  size_t max_samples, int output_rate);
std::vector<float> read_audio(const void *data, size_t bytes, size_t max_samples, int output_rate);
std::vector<float> read_audio_file(const std::string &path, size_t max_samples, int output_rate);
} // namespace tts
