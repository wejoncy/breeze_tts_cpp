#pragma once
#include <memory>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
namespace tts::text {
struct PinyinSyl { std::string text; bool raw; };
using SentenceReadings = std::vector<std::pair<std::u32string, std::vector<std::string>>>;
class PypinyinResolver {
    struct Impl;
    std::unique_ptr<Impl> impl_;
  public:
    explicit PypinyinResolver(const std::string &directory);
    ~PypinyinResolver();
    void resolve(const char32_t *, size_t, std::vector<PinyinSyl> *) const;
    uint32_t t2s(uint32_t cp) const;
    bool isNumeric(uint32_t cp) const;
    std::pair<std::string, std::string> split(std::string_view syllable) const;
    void initialsFinals(std::u32string_view word, std::vector<std::string> &initials,
                        std::vector<std::string> &finals) const;
};
} // namespace tts::text
