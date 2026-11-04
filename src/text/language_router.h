#pragma once
#include <memory>
#include <string>
#include <string_view>
#include <vector>
namespace tts::text {
// Language routing over detector, BudouX, and configured linguistic tables.
class LanguageRouter {
    struct Impl;
    std::unique_ptr<Impl> impl_;
  public:
    struct Span { std::string language; std::u32string text; };
    explicit LanguageRouter(const std::string &directory);
    ~LanguageRouter();
    std::vector<Span> split(std::u32string_view, const std::string &mode) const;
};
} // namespace tts::text
