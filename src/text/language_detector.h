#pragma once
#include <memory>
#include <string>
#include <string_view>
#include <vector>
namespace tts::text {
class LanguageDetector {
    struct Impl;
    std::unique_ptr<Impl> impl_;
  public:
    struct Candidate { std::string language; float score; };
    explicit LanguageDetector(const std::string &directory);
    ~LanguageDetector();
    std::vector<Candidate> detect(std::u32string_view, int count = 1, float threshold = 0) const;
};
} // namespace tts::text
