#pragma once
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace tts::text {
// Dictionary costs, character classes and unknown-word rules are model assets.
// Japanese and Korean share the same native Viterbi implementation.
class Morphology {
    struct Impl;
    std::unique_ptr<Impl> impl_;
  public:
    struct Token { std::string surface, feature; };
    explicit Morphology(const std::string &asset);
    ~Morphology();
    std::vector<Token> analyze(std::string_view) const;
};
} // namespace tts::text
