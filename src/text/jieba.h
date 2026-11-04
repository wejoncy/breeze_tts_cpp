#pragma once
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace tts::text {
struct SegToken { std::u32string word; std::string flag; };
// TTS adapter over cppjieba; dictionaries use its standard UTF-8 formats.
class JiebaSegmenter {
    struct Impl;
    std::unique_ptr<Impl> impl_;
  public:
    explicit JiebaSegmenter(const std::string &directory);
    ~JiebaSegmenter();
    std::vector<SegToken> segment(std::u32string_view) const;
    std::vector<std::u32string> search(std::u32string_view) const;
};
} // namespace tts::text
