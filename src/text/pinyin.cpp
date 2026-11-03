// Resolution semantics follow pypinyin (MIT, Copyright (c) 2016 mozillazg).
#include "text/pinyin.h"
#include "text/linguistic_data.h"
#include "text/utf8.h"
#include "io/json.h"
#include <algorithm>
#include <array>
#include <map>
namespace tts::text {
struct PypinyinResolver::Impl {
    struct Phrase { uint32_t offset, syllables; uint16_t length, count; };
    struct Character { uint32_t cp, syllables; uint16_t count; };
    LinguisticAsset asset;
    std::u32string codepoints;
    std::vector<Phrase> phrases;
    std::vector<Character> chars;
    std::vector<std::pair<uint32_t, uint32_t>> traditional, numeric, han_ranges;
    std::map<std::string, std::array<std::string, 2>, std::less<>> styles;
    explicit Impl(const std::string &directory) : asset(directory + "/pinyin.bin", "GSPPYX01") {
        auto cp = asset.section("cps");
        codepoints.resize(cp.remaining() / 4);
        for (auto &c : codepoints) c = cp.read<uint32_t>();
        cp.finish();
        auto words = asset.section("phrases");
        auto count = words.read<uint32_t>();
        if (count > words.remaining() / 14) throw std::invalid_argument("pronunciation phrase count");
        phrases.resize(count);
        for (auto &p : phrases) {
            p.offset = words.read<uint32_t>(); p.length = words.read<uint16_t>();
            p.syllables = words.read<uint32_t>(); p.count = words.read<uint16_t>(); words.read<uint16_t>();
            if (!p.length || p.offset > codepoints.size() || p.length > codepoints.size() - p.offset)
                throw std::invalid_argument("pronunciation phrase bounds");
            validate(p.syllables, p.count);
        }
        words.finish();
        auto characters = asset.section("chars");
        count = characters.read<uint32_t>();
        if (count > characters.remaining() / 12) throw std::invalid_argument("pronunciation character count");
        chars.resize(count);
        for (auto &c : chars) {
            c.cp = characters.read<uint32_t>(); c.syllables = characters.read<uint32_t>();
            c.count = characters.read<uint16_t>(); characters.read<uint16_t>();
            validate(c.syllables, c.count);
        }
        characters.finish();
        for (auto item : {std::pair{"t2s", &traditional}, std::pair{"numrange", &numeric}}) {
            auto entries = asset.section(item.first);
            count = entries.read<uint32_t>();
            if (count > entries.remaining() / 8) throw std::invalid_argument("pronunciation mapping count");
            item.second->resize(count);
            for (auto &pair : *item.second) {
                pair.first = entries.read<uint32_t>(); pair.second = entries.read<uint32_t>();
            }
            entries.finish();
        }
        auto data = read_json(directory + "/linguistics.json").at("pinyin");
        han_ranges = data.at("han_ranges").get<decltype(han_ranges)>();
        for (auto it = data.at("styles").begin(); it != data.at("styles").end(); ++it)
            styles.emplace(it.key(), it.value().get<std::array<std::string, 2>>());
    }
    void validate(uint32_t offset, uint16_t count) const {
        auto reader = asset.section("sylpool").slice(offset);
        if (!count) throw std::invalid_argument("empty pronunciation runs");
        while (count--) {
            auto n = reader.read<uint8_t>();
            if (!n) throw std::invalid_argument("empty pronunciation candidates");
            while (n--) reader.text(reader.read<uint8_t>());
        }
    }
    std::u32string_view key(const Phrase &p) const { return {codepoints.data() + p.offset, p.length}; }
    std::vector<Phrase>::const_iterator lower(std::u32string_view word) const {
        return std::lower_bound(phrases.begin(), phrases.end(), word,
            [&](const auto &entry, auto value) { return key(entry) < value; });
    }
    const Phrase *phrase(std::u32string_view word) const {
        auto it = lower(word);
        return it != phrases.end() && key(*it) == word ? &*it : nullptr;
    }
    bool prefix(std::u32string_view word) const {
        auto it = lower(word);
        return it != phrases.end() && key(*it).size() >= word.size() && key(*it).substr(0, word.size()) == word;
    }
    bool han(uint32_t cp) const {
        return std::any_of(han_ranges.begin(), han_ranges.end(), [&](auto range) { return cp >= range.first && cp <= range.second; });
    }
    void syllables(uint32_t offset, uint16_t count, std::vector<PinyinSyl> &out) const {
        auto reader = asset.section("sylpool").slice(offset);
        while (count--) {
            auto candidates = reader.read<uint8_t>();
            for (uint16_t i = 0; i < candidates; ++i) {
                auto text = reader.text(reader.read<uint8_t>());
                if (!i) out.push_back({std::string(text), false});
            }
        }
    }
    void single(char32_t cp, std::vector<PinyinSyl> &out) const {
        auto it = std::lower_bound(chars.begin(), chars.end(), uint32_t(cp),
            [](const auto &entry, uint32_t value) { return entry.cp < value; });
        if (it != chars.end() && it->cp == cp) syllables(it->syllables, it->count, out);
        else out.push_back({encode(std::u32string_view(&cp, 1)), true});
    }
    void resolve(std::u32string_view input, std::vector<PinyinSyl> &out) const {
        out.clear();
        for (size_t start = 0; start < input.size();) {
            bool is_han = han(input[start]);
            size_t end = start + 1;
            while (end < input.size() && han(input[end]) == is_han) ++end;
            if (!is_han) out.push_back({encode(input.substr(start, end - start)), true});
            else for (size_t first = start; first < end;) {
                // Forward maximum matching only accepts complete phrase keys.
                size_t length = 1;
                const Phrase *chosen = nullptr;
                for (size_t last = first + 1; last <= end; ++last) {
                    auto word = input.substr(first, last - first);
                    if (!prefix(word)) break;
                    if (auto *candidate = phrase(word)) { chosen = candidate; length = word.size(); }
                }
                if (chosen) syllables(chosen->syllables, chosen->count, out);
                else single(input[first], out);
                first += length;
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
