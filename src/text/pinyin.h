#pragma once
#include <memory>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
namespace tts::text {
struct PinyinSyl { std::string text; bool raw; };
class PinyinResolver {
  public:
    virtual ~PinyinResolver() = default;
    virtual void resolve(const char32_t *, size_t, std::vector<PinyinSyl> *) const = 0;
    virtual bool resolveSentenceTone3(const char32_t *, size_t, std::vector<std::string> *) const { return false; }
    virtual bool isNumeric(uint32_t cp) const { return cp >= U'0' && cp <= U'9'; }
};
class PypinyinResolver final : public PinyinResolver {
    struct Impl;
    std::unique_ptr<Impl> impl_;
  public:
    explicit PypinyinResolver(const std::string &directory);
    ~PypinyinResolver();
    void resolve(const char32_t *, size_t, std::vector<PinyinSyl> *) const override;
    uint32_t t2s(uint32_t cp) const;
    bool isNumeric(uint32_t cp) const override;
    std::pair<std::string, std::string> split(std::string_view syllable) const;
    void initialsFinals(std::u32string_view word, std::vector<std::string> &initials,
                        std::vector<std::string> &finals) const;
};
} // namespace tts::text
