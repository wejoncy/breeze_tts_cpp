#include "text/regex.h"
#include <memory>
#include <stdexcept>
#include <utility>
extern "C" {
void *hf_regex_new(const char *, size_t, char *, size_t);
void hf_regex_free(void *);
int hf_regex_find(const void *, const char *, size_t, tts::text::MatchSpan **, size_t *, char *, size_t);
void hf_regex_spans_free(tts::text::MatchSpan *, size_t);
}
namespace tts::text {
Regex::Regex(std::string_view pattern) {
    char error[1024]{};
    impl_ = hf_regex_new(pattern.empty() ? "" : pattern.data(), pattern.size(), error, sizeof(error));
    if (!impl_) throw std::invalid_argument(error);
}
Regex::~Regex() { hf_regex_free(impl_); }
Regex::Regex(Regex &&other) noexcept : impl_(std::exchange(other.impl_, nullptr)) {}
Regex &Regex::operator=(Regex &&other) noexcept {
    if (this != &other) { hf_regex_free(impl_); impl_ = std::exchange(other.impl_, nullptr); }
    return *this;
}
std::vector<MatchSpan> Regex::find(std::string_view text) const {
    char error[1024]{};
    MatchSpan *spans = nullptr;
    size_t count = 0;
    if (hf_regex_find(impl_, text.empty() ? "" : text.data(), text.size(), &spans, &count, error, sizeof(error)))
        throw std::runtime_error(error);
    auto release = [count](MatchSpan *p) { hf_regex_spans_free(p, count); };
    std::unique_ptr<MatchSpan, decltype(release)> owned(spans, release);
    return count ? std::vector<MatchSpan>(spans, spans + count) : std::vector<MatchSpan>{};
}
} // namespace tts::text
