#pragma once
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace tts::text {
class PypinyinResolver;
// ToJyutping's complete phrase dictionary and longest-match pronunciation,
// followed by the GPT-SoVITS Cantonese phone mapping. Runs on the scheduler.
class CantonesePhonemizer {
    struct Impl;
    std::unique_ptr<Impl> impl_;
  public:
    CantonesePhonemizer(const std::string &directory, const PypinyinResolver &);
    ~CantonesePhonemizer();
    std::u32string normalize(std::u32string_view) const;
    std::vector<int> pronounce(std::u32string_view normalized) const;
};
} // namespace tts::text
