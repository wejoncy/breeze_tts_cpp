#pragma once
#include <memory>
#include <string>
#include <string_view>
#include <vector>
namespace tts::text {
// Official g2pk2 rules over the shared native dictionary/Viterbi analyzer.
class KoreanPhonemizer {
    struct Impl;
    std::unique_ptr<Impl> impl_;
  public:
    explicit KoreanPhonemizer(const std::string &directory);
    ~KoreanPhonemizer();
    std::vector<int> pronounce(std::u32string_view) const;
};
} // namespace tts::text
