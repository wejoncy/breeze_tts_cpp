// Korean rules: g2pk2 (Apache-2.0) and GPT-SoVITS (MIT). Native adaptation.
#include "text/korean.h"
#include "text/morphology.h"
#include "text/regex.h"
#include "text/utf8.h"
#include "io/json.h"
#include <algorithm>
#include <map>
#include <optional>
#include <set>

namespace tts::text {
namespace {
using Text = std::u32string;
void replace(Text &text, const Text &from, const Text &to) {
    if (from.empty()) throw std::invalid_argument("empty Korean replacement");
    for (size_t cursor = 0; (cursor = text.find(from, cursor)) != Text::npos; cursor += to.size())
        text.replace(cursor, from.size(), to);
}
bool syllable(char32_t cp) { return cp >= 0xac00 && cp <= 0xd7a3; }
Text decompose(std::u32string_view text) {
    Text out;
    for (auto cp : text) {
        if (!syllable(cp)) { out += cp; continue; }
        auto value = cp - 0xac00;
        out += 0x1100 + value / 588;
        out += 0x1161 + value % 588 / 28;
        if (value % 28) out += 0x11a7 + value % 28;
    }
    return out;
}
struct Rewrite {
    Regex pattern;
    struct Piece { std::string literal; int mode = -1; size_t a = 0, b = 0; };
    std::vector<Piece> pieces;
    explicit Rewrite(const Json &data) : pattern(data.at(0).get<std::string>()) {
        for (const auto &part : data.at(1)) {
            if (part.is_string()) pieces.push_back({part.get<std::string>(), -1, 0, 0});
            else {
                int mode = part.at(0);
                if (mode < 0 || mode > 2 || part.size() != 3)
                    throw std::invalid_argument("invalid Korean capture slice");
                pieces.push_back({{}, mode, part.at(1), part.at(2)});
            }
        }
    }
    void apply(std::string &text) const {
        auto matches = pattern.find(text);
        if (matches.empty()) return;
        std::string out;
        out.reserve(text.size());
        size_t cursor = 0;
        for (auto match : matches) {
            out.append(text, cursor, match.start - cursor);
            auto matched = std::string_view(text).substr(match.start, match.end - match.start);
            std::vector<size_t> offsets;
            decode(matched, &offsets);
            const size_t count = offsets.size() - 1;
            for (const auto &piece : pieces) {
                if (piece.mode < 0) { out += piece.literal; continue; }
                size_t start, length;
                if (piece.mode == 0) { start = piece.a; length = piece.b; }
                else if (piece.mode == 1) {
                    if (piece.a > count || piece.b > count - piece.a)
                        throw std::invalid_argument("Korean capture outside match");
                    start = count - piece.a - piece.b; length = piece.b;
                } else {
                    start = piece.a;
                    if (start > count || piece.b > count - start)
                        throw std::invalid_argument("Korean capture outside match");
                    length = count - start - piece.b;
                }
                if (start > count || length > count - start)
                    throw std::invalid_argument("Korean capture outside match");
                out.append(matched.substr(offsets[start], offsets[start + length] - offsets[start]));
            }
            cursor = match.end;
        }
        out.append(text, cursor, text.size() - cursor);
        text.swap(out);
    }
};
void apply(const std::vector<Rewrite> &rules, std::string &text) {
    for (const auto &rule : rules) rule.apply(text);
}
std::map<char32_t, Text> character_map(const Json &data) {
    std::map<char32_t, Text> result;
    for (const auto &[key, value] : data.items()) {
        auto cp = decode(key);
        if (cp.size() != 1) throw std::invalid_argument("Korean map key is not a character");
        result.emplace(cp.front(), decode(value.get<std::string>()));
    }
    return result;
}
} // namespace
struct KoreanPhonemizer::Impl {
    Morphology morphology;
    std::vector<Rewrite> idioms, special, regular, links;
    Regex remove_annotations{"/[PJEB]"};
    std::map<char32_t, Text> latin, compatibility, replacements;
    std::vector<std::pair<Text, Text>> divided;
    std::vector<std::pair<char32_t, char32_t>> decimal;
    std::vector<Text> modifiers, tens;
    Text digits, scales, twenty, bound_nouns;
    std::map<char32_t, int> symbols;
    int unknown;
    explicit Impl(const std::string &directory)
        : morphology(directory + "/korean_mecab") {
        auto data = read_json(directory + "/korean.json");
        for (const auto &item : {std::pair{"idioms", &idioms}, {"special", &special},
                                 {"regular", &regular}, {"links", &links}})
            for (const auto &rule : data.at(item.first)) item.second->emplace_back(rule);
        latin = character_map(data.at("latin"));
        compatibility = character_map(data.at("compatibility"));
        replacements = character_map(data.at("replacements"));
        for (const auto &item : data.at("divided"))
            divided.emplace_back(decode(item.at(0).get<std::string>()),
                                 decode(item.at(1).get<std::string>()));
        for (const auto &item : data.at("decimal")) decimal.emplace_back(item.at(0), item.at(1));
        for (const auto &s : data.at("modifiers")) modifiers.push_back(decode(s.get<std::string>()));
        for (const auto &s : data.at("tens")) tens.push_back(decode(s.get<std::string>()));
        digits = decode(data.at("digits").get<std::string>());
        scales = decode(data.at("scales").get<std::string>());
        twenty = decode(data.at("twenty").get<std::string>());
        bound_nouns = decode(data.at("bound_nouns").get<std::string>());
        auto vocabulary = read_json(directory + "/linguistics.json").at("symbols");
        for (size_t i = 0; i < vocabulary.size(); ++i) {
            auto s = decode(vocabulary[i].get<std::string>());
            if (s.size() == 1) symbols.emplace(s.front(), int(i));
        }
        unknown = symbols.at(U'停');
        if (modifiers.size() != 9 || tens.size() != 9 || digits.size() != 10 || scales.size() != 15)
            throw std::invalid_argument("invalid Korean numeral tables");
    }
    bool digit(char32_t cp) const {
        auto found = std::upper_bound(decimal.begin(), decimal.end(), cp,
            [](char32_t value, auto range) { return value < range.first; });
        return found != decimal.begin() && cp <= (--found)->second;
    }
    Text annotate(const Text &text) const {
        auto words = morphology.analyze(encode(text));
        Text joined, tags;
        for (const auto &word : words) {
            auto token = decode(word.surface);
            joined += token;
            auto tag = word.feature.substr(0, word.feature.find(','));
            auto plus = tag.find_last_of('+');
            if (plus != std::string::npos) tag.erase(0, plus + 1);
            if (token.empty() || tag.empty()) throw std::invalid_argument("empty Korean morphology token");
            tags.append(token.size() - 1, U'_');
            tags += tag == "NNBC" ? U'B' : char32_t(tag.front());
        }
        Text compact(text);
        compact.erase(std::remove(compact.begin(), compact.end(), U' '), compact.end());
        if (compact != joined) return text;
        Text out;
        size_t token = 0;
        for (auto cp : text) {
            out += cp;
            if (cp == U' ') continue;
            auto tag = tags.at(token++);
            auto final = decompose(Text(1, cp)).back();
            if (cp == U'의' && tag == U'J') out += U"/J";
            else if (tag == U'E' && final == U'ᆯ') out += U"/E";
            else if (tag == U'V' && std::u32string_view(U"ᆫᆬᆷᆱᆰᆲᆴ").find(final) != Text::npos) out += U"/P";
            else if (tag == U'B') out += U"/B";
        }
        return out;
    }
    Text number(Text num, bool sino) const {
        num.erase(std::remove(num.begin(), num.end(), U','), num.end());
        if (num == U"0") return Text(1, digits.front());
        if (!sino && num == U"20") return twenty;
        std::vector<Text> parts;
        std::optional<Text> name;
        for (size_t cursor = 0; cursor < num.size(); ++cursor) {
            auto cp = num[cursor];
            size_t place = num.size() - cursor - 1;
            const bool known = cp >= U'1' && cp <= U'9';
            Text spoken = known ? Text(1, digits.at(cp - U'0')) : Text{};
            if (place == 0) name = sino ? spoken : known ? modifiers.at(cp - U'1') : Text{};
            else if (place == 1) name = sino ? spoken + scales.front()
                                            : known ? tens.at(cp - U'1') : Text{};
            if (cp == U'0') {
                bool empty = true;
                for (size_t i = parts.size() > 3 ? parts.size() - 3 : 0; i < parts.size(); ++i)
                    if (!parts[i].empty()) empty = false;
                if (place % 4 || empty) { parts.emplace_back(); continue; }
            }
            if (place >= 2 && place <= scales.size()) name = spoken + scales[place - 1];
            if (!name) throw std::invalid_argument("Korean numeral exceeds upstream supported range");
            if (place >= 1 && place <= 7 && (sino || place > 1) &&
                name->size() == 2 && name->front() == digits[1]) name->erase(0, 1);
            parts.push_back(*name);
        }
        Text out;
        for (const auto &part : parts) out += part;
        return out;
    }
    Text numbers(const Text &text) const {
        Text out;
        for (size_t i = 0; i < text.size();) {
            size_t last = i;
            if (digit(text[i])) {
                while (last < text.size() && (digit(text[last]) || text[last] == U',')) ++last;
                for (size_t split = last; split > i; --split) {
                    size_t end = split;
                    while (end < text.size() && text[end] >= U'ㄱ' && text[end] <= U'힣') ++end;
                    if (end == split || text.substr(end, 2) != U"/B") continue;
                    auto noun = text.substr(split, end - split);
                    out += number(text.substr(i, split - i), bound_nouns.find(noun) == Text::npos);
                    out += noun + U"/B";
                    i = end + 2;
                    break;
                }
                if (i > last) continue;
            }
            auto cp = text[i++];
            out += cp >= U'0' && cp <= U'9' ? digits[cp - U'0'] : cp;
        }
        return out;
    }
    std::vector<int> pronounce(std::u32string_view text) const {
        Text spelled;
        for (auto cp : text) {
            auto found = latin.find(cp);
            if (found != latin.end()) spelled += found->second;
            else spelled += cp;
        }
        std::string value = encode(spelled);
        apply(idioms, value);
        value = encode(decompose(numbers(annotate(decode(value)))));
        apply(special, value);
        auto annotations = remove_annotations.find(value);
        for (auto it = annotations.rbegin(); it != annotations.rend(); ++it)
            value.erase(it->start, it->end - it->start);
        apply(regular, value);
        apply(links, value);
        auto jamo = decode(value);
        Text out;
        auto lead = [](char32_t cp) { return cp >= 0x1100 && cp <= 0x1112; };
        auto vowel = [](char32_t cp) { return cp >= 0x1161 && cp <= 0x1175; };
        // compose() inserts the silent onset before isolated vowels. Its modern
        // syllable composition is immediately reversed by divide_hangul's h2j.
        for (size_t i = 0; i < jamo.size();) {
            if (!i && vowel(jamo[i])) { out += U'ᄋ'; out += jamo[i++]; }
            else if (!lead(jamo[i]) && i + 1 < jamo.size() && vowel(jamo[i + 1])) {
                out += jamo[i++]; out += U'ᄋ'; out += jamo[i++];
            } else out += jamo[i++];
        }
        Text hcj;
        for (auto cp : out) {
            auto found = compatibility.find(cp);
            if (found != compatibility.end()) hcj += found->second;
            else {
                if ((cp >= 0x1100 && cp <= 0x11ff) || (cp >= 0xa960 && cp <= 0xa97c) ||
                    (cp >= 0xd7b0 && cp <= 0xd7c6) || (cp >= 0xd7cb && cp <= 0xd7fb))
                    throw std::invalid_argument("Korean jamo is not assigned in the source compatibility table");
                hcj += cp;
            }
        }
        for (const auto &pair : divided) replace(hcj, pair.first, pair.second);
        for (size_t i = 0; i + 4 < hcj.size();) {
            if ((hcj.substr(i, 3) == U"ㅇㅡㄹ" || hcj.substr(i, 3) == U"ㄹㅡㄹ") &&
                hcj[i + 3] == U' ' && hcj[i + 4] == U'ㄹ') { hcj[i + 4] = U'ㄴ'; i += 5; }
            else ++i;
        }
        if (!hcj.empty() && hcj.back() >= 0x3131 && hcj.back() <= 0x3163) hcj += U'.';
        std::vector<int> phones;
        for (auto cp : hcj) {
            Text ph(1, cp);
            if (auto found = replacements.find(cp); found != replacements.end()) ph = found->second;
            auto found = ph.size() == 1 ? symbols.find(ph.front()) : symbols.end();
            phones.push_back(found == symbols.end() ? unknown : found->second);
        }
        return phones;
    }
};
KoreanPhonemizer::KoreanPhonemizer(const std::string &directory) : impl_(std::make_unique<Impl>(directory)) {}
KoreanPhonemizer::~KoreanPhonemizer() = default;
std::vector<int> KoreanPhonemizer::pronounce(std::u32string_view text) const { return impl_->pronounce(text); }
} // namespace tts::text
