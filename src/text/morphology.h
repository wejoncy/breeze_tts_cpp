#pragma once
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace tts::text {
// Japanese and Korean share the native MeCab API with checkpoint dictionaries.
class Morphology {
    struct Impl;
    std::unique_ptr<Impl> impl_;
  public:
    struct Token { std::string surface, feature; };
    explicit Morphology(const std::string &directory, const std::string &user_dictionary = {});
    ~Morphology();
    std::vector<Token> analyze(std::string_view) const;
};
} // namespace tts::text
