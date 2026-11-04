#pragma once
#include <memory>
#include <string>
#include <string_view>
#include <vector>
namespace tts::text {
class JapanesePhonemizer {
    struct Impl;
    std::unique_ptr<Impl> impl_;
  public:
    explicit JapanesePhonemizer(const std::string &directory);
    ~JapanesePhonemizer();
    std::u32string normalize(std::u32string_view) const;
    std::vector<int> pronounce(std::u32string_view normalized) const;
};
} // namespace tts::text
