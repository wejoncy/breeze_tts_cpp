#pragma once
#include <string>
#include <vector>

namespace tts::text {
// GPT-SoVITS cut0..cut5, followed by short-fragment joining and bounded
// Unicode splitting. Empty method preserves a single input verbatim.
std::vector<std::string> segment_text(const std::string &, const std::string &method,
                                    bool english, size_t max_characters);
} // namespace tts::text
