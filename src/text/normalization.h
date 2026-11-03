#pragma once
#include <memory>
#include <string>
#include <string_view>
#include <vector>
namespace tts::text {
class PypinyinResolver;
class TextNormalizer {
    struct Impl;
    std::unique_ptr<Impl> impl_;
  public:
    TextNormalizer(const std::string &directory, const PypinyinResolver &dictionary);
    ~TextNormalizer();
    std::vector<std::u32string> normalize(std::u32string_view text) const;
};
} // namespace tts::text
