#pragma once
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace tts::text {
// CPU linguistic work. Assets contain the dictionaries, POS model, word counts
// and the small grapheme-to-phoneme GRU; inference does not call Python.
class EnglishPhonemizer {
    struct Impl;
    std::unique_ptr<Impl> impl_;
  public:
    explicit EnglishPhonemizer(const std::string &directory);
    ~EnglishPhonemizer();
    std::string normalize(std::string_view) const;
    std::vector<int> pronounce(const std::string &normalized) const;
};
} // namespace tts::text
