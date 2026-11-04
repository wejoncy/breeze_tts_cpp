#pragma once
#include <cstddef>
#include <string_view>
#include <vector>
namespace tts::text {
struct MatchSpan { size_t start, end; };
class Regex {
    void *impl_;
  public:
    explicit Regex(std::string_view pattern);
    ~Regex();
    Regex(const Regex &) = delete;
    Regex &operator=(const Regex &) = delete;
    Regex(Regex &&other) noexcept;
    Regex &operator=(Regex &&other) noexcept;
    std::vector<MatchSpan> find(std::string_view text) const;
};
} // namespace tts::text
