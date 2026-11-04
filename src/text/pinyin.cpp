// Resolution semantics follow pypinyin (MIT, Copyright (c) 2016 mozillazg).
#include "text/pinyin.h"
#include "text/lexicon.h"
#include "text/utf8.h"
#include "io/json.h"
#include <algorithm>
#include <array>
#include <map>
namespace tts::text {
struct PypinyinResolver::Impl {
    Lexicon<std::string> phrases;
    std::map<char32_t, std::string> characters;
    std::vector<std::pair<uint32_t, uint32_t>> traditional, numeric, han_ranges;
    std::map<std::string, std::array<std::string, 2>, std::less<>> styles;
    explicit Impl(const std::string &directory) {
        const auto dictionary = read_json(directory + "/pinyin.json");
        phrases = load_lexicon<std::string>(dictionary.at("phrases"));
        for (const auto &[word, readings] : phrases)
            if (readings.empty() || std::any_of(readings.begin(), readings.end(),
                    [](const auto &reading) { return reading.empty(); }))
                throw std::invalid_argument("empty pronunciation phrase reading");
        for (const auto &[character, reading] : dictionary.at("characters").items()) {
            auto codepoints = decode(character);
            auto value = reading.get<std::string>();
            if (codepoints.size() != 1 || value.empty())
                throw std::invalid_argument("invalid pronunciation character reading");
            characters.emplace(codepoints.front(), std::move(value));
        }
        traditional = dictionary.at("traditional").get<decltype(traditional)>();
        numeric = dictionary.at("numeric").get<decltype(numeric)>();
        const auto data = read_json(directory + "/linguistics.json").at("pinyin");
        han_ranges = data.at("han_ranges").get<decltype(han_ranges)>();
        for (const auto &[reading, values] : data.at("styles").items())
            styles.emplace(reading, values.get<std::array<std::string, 2>>());
    }
    bool han(char32_t codepoint) const {
        return std::any_of(han_ranges.begin(), han_ranges.end(),
            [&](auto range) { return codepoint >= range.first && codepoint <= range.second; });
    }
    void resolve(std::u32string_view input, std::vector<PinyinSyl> &output) const {
        output.clear();
        for (size_t start = 0; start < input.size();) {
            const bool is_han = han(input[start]);
            size_t end = start + 1;
            while (end < input.size() && han(input[end]) == is_han) ++end;
            if (!is_han) output.push_back({encode(input.substr(start, end - start)), true});
            else for (size_t first = start; first < end;) {
                auto selected = longest_match(phrases, input.substr(first, end - first));
                if (selected != phrases.end()) {
                    for (const auto &reading : selected->second) output.push_back({reading, false});
                    first += selected->first.size();
                } else {
                    auto character = characters.find(input[first]);
                    output.push_back(character == characters.end()
                        ? PinyinSyl{encode(input.substr(first, 1)), true}
                        : PinyinSyl{character->second, false});
                    ++first;
                }
            }
            start = end;
        }
    }
};
PypinyinResolver::PypinyinResolver(const std::string &directory) : impl_(std::make_unique<Impl>(directory)) {}
PypinyinResolver::~PypinyinResolver() = default;
void PypinyinResolver::resolve(const char32_t *word, size_t size, std::vector<PinyinSyl> *out) const {
    impl_->resolve({word, size}, *out);
}
uint32_t PypinyinResolver::t2s(uint32_t cp) const {
    const auto &entries = impl_->traditional;
    auto it = std::lower_bound(entries.begin(), entries.end(), cp, [](auto pair, uint32_t value) { return pair.first < value; });
    return it != entries.end() && it->first == cp ? it->second : cp;
}
bool PypinyinResolver::isNumeric(uint32_t cp) const {
    const auto &ranges = impl_->numeric;
    auto it = std::upper_bound(ranges.begin(), ranges.end(), cp, [](uint32_t value, auto range) { return value < range.first; });
    return it != ranges.begin() && cp <= std::prev(it)->second;
}
std::pair<std::string, std::string> PypinyinResolver::split(std::string_view syllable) const {
    auto it = impl_->styles.find(syllable);
    if (it == impl_->styles.end()) throw std::invalid_argument("pronunciation missing from exported styles: " + std::string(syllable));
    return {it->second[0], it->second[1]};
}
void PypinyinResolver::initialsFinals(std::u32string_view word, std::vector<std::string> &initials, std::vector<std::string> &finals) const {
    std::vector<PinyinSyl> syllables;
    impl_->resolve(word, syllables);
    initials.clear(); finals.clear();
    for (const auto &s : syllables) {
        auto style = s.raw ? std::pair{s.text, s.text} : split(s.text);
        initials.push_back(std::move(style.first)); finals.push_back(std::move(style.second));
    }
}
} // namespace tts::text
