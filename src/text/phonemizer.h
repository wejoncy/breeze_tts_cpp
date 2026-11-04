#pragma once
#include "text/pinyin.h"
#include <memory>
namespace tts::text {
struct Phonemes { std::vector<int> phones, word2ph; };
class ChinesePhonemizer {
    struct Impl;
    std::unique_ptr<Impl> impl_;
  public:
    explicit ChinesePhonemizer(const std::string &directory);
    ~ChinesePhonemizer();
    const PypinyinResolver &dictionary() const;
    std::u32string normalize(std::u32string_view input) const;
    std::vector<std::u32string> segments(std::u32string_view normalized) const;
    Phonemes pronounce(std::u32string_view normalized, const SentenceReadings &) const;
};
} // namespace tts::text
