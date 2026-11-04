// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
// Copyright (c) 2026 tts.cpp contributors.
// SPDX-License-Identifier: Apache-2.0
// Native implementation of the ordered Chinese verbalization rules in
// GPT-SoVITS/text/zh_normalization. Rule patterns and linguistic tables are
// loaded from model assets; regex matching uses the existing tokenizer bridge.
#include "text/normalization.h"
#include "text/pinyin.h"
#include "text/regex.h"
#include "text/utf8.h"
#include "io/json.h"
#include <algorithm>
#include <array>
#include <map>
#include <stdexcept>

namespace tts::text {
namespace {
using Text = std::u32string;
using View = std::u32string_view;
bool digit(char32_t c) { return c >= U'0' && c <= U'9'; }
bool space(char32_t c) {
    return (c >= U'\t' && c <= U'\r') || c == U' ' || c == 0x85 || c == 0xa0 ||
           c == 0x1680 || (c >= 0x2000 && c <= 0x200a) || c == 0x2028 || c == 0x2029 ||
           c == 0x202f || c == 0x205f || c == 0x3000;
}
View trim(View s) {
    while (!s.empty() && space(s.front())) s.remove_prefix(1);
    while (!s.empty() && space(s.back())) s.remove_suffix(1);
    return s;
}
View nonzero(View s) {
    while (!s.empty() && s.front() == U'0') s.remove_prefix(1);
    return s;
}
size_t number_end(View s, size_t start = 0) {
    size_t end = start;
    while (end < s.size() && digit(s[end])) ++end;
    if (end < s.size() && s[end] == U'.') {
        ++end;
        while (end < s.size() && digit(s[end])) ++end;
    }
    return end;
}
std::vector<View> split(View s, char32_t separator) {
    std::vector<View> parts;
    size_t start = 0;
    for (;;) {
        auto end = s.find(separator, start);
        parts.push_back(s.substr(start, end == View::npos ? end : end - start));
        if (end == View::npos) return parts;
        start = end + 1;
    }
}
void replace_all(Text &s, View from, View to) {
    if (from.empty()) throw std::invalid_argument("empty normalization replacement");
    size_t offset = 0;
    while ((offset = s.find(from, offset)) != Text::npos) {
        s.replace(offset, from.size(), to);
        offset += to.size();
    }
}
Text filtered(View s, View remove) {
    Text out;
    out.reserve(s.size());
    for (char32_t c : s)
        if (remove.find(c) == View::npos) out.push_back(c);
    return out;
}
} // namespace
struct TextNormalizer::Impl {
    struct Rule { std::string kind; Regex pattern; };
    const PypinyinResolver &dictionary;
    std::vector<Rule> rules;
    std::unique_ptr<Regex> splitter;
    std::vector<std::pair<Text, Text>> measures, post;
    std::array<Text, 10> digits;
    std::map<size_t, Text> units;
    std::map<char32_t, Text> arithmetic, powers;
    Text before, after, alternate_one, quantifier_two;
    Impl(const std::string &directory, const PypinyinResolver &d) : dictionary(d) {
        const auto cfg = read_json(directory + "/linguistics.json").at("normalization");
        splitter = std::make_unique<Regex>(cfg.at("splitter").get<std::string>());
        for (const auto &r : cfg.at("rules")) {
            auto kind = r.at(0).get<std::string>();
            auto pattern = r.at(1).get<std::string>();
            rules.push_back({kind, Regex(kind == "range" ? "(?x)" + pattern : pattern)});
        }
        auto pairs = [](const Json &rows) {
            std::vector<std::pair<Text, Text>> values;
            for (const auto &r : rows)
                values.emplace_back(decode(r.at(0).get<std::string>()),
                                    decode(r.at(1).get<std::string>()));
            return values;
        };
        measures = pairs(cfg.at("measure"));
        post = pairs(cfg.at("post"));
        for (size_t i = 0; i < digits.size(); ++i)
            digits[i] = decode(cfg.at("digits").at(std::to_string(i)).get<std::string>());
        for (auto it = cfg.at("units").begin(); it != cfg.at("units").end(); ++it)
            units.emplace(std::stoul(it.key()), decode(it.value().get<std::string>()));
        auto char_map = [](const Json &values) {
            std::map<char32_t, Text> out;
            for (auto it = values.begin(); it != values.end(); ++it) {
                auto key = decode(it.key());
                if (key.size() != 1) throw std::invalid_argument("normalization character map key");
                out.emplace(key[0], decode(it.value().get<std::string>()));
            }
            return out;
        };
        arithmetic = char_map(cfg.at("asmd"));
        powers = char_map(cfg.at("power"));
        before = decode(cfg.at("strip_before").get<std::string>());
        after = decode(cfg.at("strip_after").get<std::string>());
        alternate_one = decode(cfg.at("alt_one").get<std::string>());
        quantifier_two = decode(cfg.at("quantifier_two").get<std::string>());
        if (units.empty() || units.begin()->first != 1 || digits[0].empty())
            throw std::invalid_argument("invalid normalization number tables");
    }
    Text digit_words(View s, bool alternate = false) const {
        Text out;
        for (char32_t c : s) {
            if (!digit(c)) throw std::invalid_argument("unsupported numeric character");
            out += alternate && c == U'1' ? alternate_one : digits[c - U'0'];
        }
        return out;
    }
    Text value(View original) const {
        auto stripped = nonzero(original);
        if (stripped.empty()) return {};
        if (stripped.size() == 1)
            return (stripped.size() < original.size() ? digits[0] : Text{}) + digit_words(stripped);
        auto unit = units.lower_bound(stripped.size());
        if (unit == units.begin()) throw std::invalid_argument("missing numeric place unit");
        --unit;
        size_t suffix = unit->first;
        return value(original.substr(0, original.size() - suffix)) + unit->second +
               value(original.substr(original.size() - suffix));
    }
    Text cardinal(View s) const {
        if (s.empty()) return {};
        s = nonzero(s);
        if (s.empty()) return digits[0];
        Text out = value(s);
        Text abbreviated = digits[1] + units.at(1);
        if (out.compare(0, abbreviated.size(), abbreviated) == 0)
            out.erase(0, digits[1].size());
        return out;
    }
    Text number(View s) const {
        auto point = s.find(U'.');
        if (point == View::npos) return cardinal(s);
        if (s.find(U'.', point + 1) != View::npos)
            throw std::invalid_argument("multiple decimal points");
        Text out = cardinal(s.substr(0, point));
        auto decimal = s.substr(point + 1);
        bool trailing_zero = !decimal.empty() && decimal.back() == U'0';
        while (!decimal.empty() && decimal.back() == U'0') decimal.remove_suffix(1);
        Text spoken = digit_words(decimal);
        if (trailing_zero) spoken += digits[0];
        if (!spoken.empty()) out = (out.empty() ? digits[0] : out) + U"点" + spoken;
        return out;
    }
    Text signed_number(View s) const {
        if (!s.empty() && s.front() == U'-') return Text(U"负") + number(s.substr(1));
        return number(s);
    }
    Text time_number(View s) const {
        return (s.front() == U'0' ? digits[0] : Text{}) + number(nonzero(s));
    }
    Text time(View s, bool force_half = false) const {
        auto parts = split(s, U':');
        if (parts.size() < 2 || parts.size() > 3) throw std::invalid_argument("invalid time match");
        Text out = number(parts[0]) + U"点";
        if (!nonzero(parts[1]).empty())
            out += force_half ? Text(U"半") : time_number(parts[1]) + U"分";
        if (parts.size() == 3 && !nonzero(parts[2]).empty()) out += time_number(parts[2]) + U"秒";
        return out;
    }
    Text verbalize(const std::string &kind, View s) const {
        if (kind == "number" || kind == "decimal" || kind == "negative") return signed_number(s);
        if (kind == "default") return digit_words(s, true);
        if (kind == "version") {
            Text out;
            for (char32_t c : s) out += c == U'.' ? Text(U"点") : digit_words(View(&c, 1));
            return out;
        }
        if (kind == "date" || kind == "date2") {
            Text out;
            size_t pos = 0, field = 0;
            while (pos < s.size()) {
                if (!digit(s[pos])) { ++pos; continue; }
                size_t end = pos;
                while (end < s.size() && digit(s[end])) ++end;
                auto part = s.substr(pos, end - pos);
                if (field == 0) out += digit_words(part) + U"年";
                else if (kind == "date") out += cardinal(part) + s[end];
                else out += cardinal(part) + (field == 1 ? U'月' : U'日');
                ++field;
                pos = end;
            }
            return out;
        }
        if (kind == "time" || kind == "time_range") {
            auto separator = s.find_first_of(U"~-");
            auto first = s.substr(0, separator);
            auto minute = split(first, U':').at(1);
            bool half = minute == U"30";
            Text out = time(first, half);
            // Preserve the upstream range rule's choice of the first minute
            // when deciding whether the second timestamp uses 'half'.
            if (separator != View::npos) out += U"至" + time(s.substr(separator + 1), half);
            return out;
        }
        if (kind == "to_range") { Text out(s); replace_all(out, U"~", U"至"); return out; }
        if (kind == "temperature") {
            bool negative = s.front() == U'-';
            if (negative) s.remove_prefix(1);
            auto end = number_end(s);
            return (negative ? Text(U"零下") : Text{}) + number(s.substr(0, end)) + U"度";
        }
        if (kind == "frac") {
            bool negative = s.front() == U'-';
            if (negative) s.remove_prefix(1);
            auto slash = s.find(U'/');
            return (negative ? Text(U"负") : Text{}) + number(s.substr(slash + 1)) +
                   U"分之" + number(s.substr(0, slash));
        }
        if (kind == "percentage") {
            bool negative = s.front() == U'-';
            if (negative) s.remove_prefix(1);
            s.remove_suffix(1);
            return (negative ? Text(U"负") : Text{}) + U"百分之" + number(s);
        }
        if (kind == "mobile" || kind == "phone" || kind == "national_phone") {
            Text out;
            if (kind == "mobile" && s.front() == U'+') s.remove_prefix(1);
            size_t begin = 0;
            for (size_t i = 0; i <= s.size(); ++i)
                if (i == s.size() || (kind == "mobile" ? space(s[i]) : s[i] == U'-')) {
                    if (i > begin || kind != "mobile") {
                        if (!out.empty()) out += U'，';
                        out += digit_words(s.substr(begin, i - begin), true);
                    }
                    begin = i + 1;
                }
            return out;
        }
        if (kind == "range") {
            auto end = number_end(s, s.front() == U'-' ? 1 : 0);
            return signed_number(s.substr(0, end)) + U"到" + signed_number(s.substr(end + 1));
        }
        if (kind == "quantifier") {
            size_t end = 0;
            while (end < s.size() && digit(s[end])) ++end;
            Text out = number(s.substr(0, end));
            if (out == digits[2]) out = quantifier_two;
            Text suffix(s.substr(end));
            if (!suffix.empty() && suffix.front() == U'+') suffix.replace(0, 1, U"多");
            return out + suffix;
        }
        if (kind == "asmd") {
            for (size_t i = 1; i < s.size(); ++i) {
                auto op = arithmetic.find(s[i]);
                if (op != arithmetic.end()) return Text(s.substr(0, i)) + op->second + Text(s.substr(i + 1));
            }
            throw std::invalid_argument("arithmetic match has no binary operator");
        }
        if (kind == "power") {
            Text out = U"的";
            for (char32_t c : s) out += powers.at(c);
            return out + U"次方";
        }
        throw std::invalid_argument("unknown normalization rule: " + kind);
    }
    bool apply(const Rule &rule, Text &s) const {
        auto input = encode(s);
        auto matches = rule.pattern.find(input);
        if (matches.empty()) return false;
        std::string out;
        size_t offset = 0;
        for (const auto &match : matches) {
            if (match.start < offset || match.end < match.start || match.end > input.size())
                throw std::invalid_argument("invalid normalization regex span");
            out.append(input, offset, match.start - offset);
            out += encode(verbalize(rule.kind, decode(std::string_view(input).substr(
                match.start, match.end - match.start))));
            offset = match.end;
        }
        out.append(input, offset, input.size() - offset);
        if (out == input) throw std::invalid_argument("normalization rewrite made no progress");
        s = decode(out);
        return true;
    }
    Text sentence(View source) const {
        Text s;
        s.reserve(source.size());
        for (auto c : source) {
            c = dictionary.t2s(c);
            if ((c >= 0xff21 && c <= 0xff3a) || (c >= 0xff41 && c <= 0xff5a) ||
                (c >= 0xff10 && c <= 0xff19)) c -= 0xfee0;
            else if (c == 0x3000) c = U' ';
            s.push_back(c);
        }
        for (const auto &rule : rules) {
            if (rule.kind == "measure") {
                for (const auto &p : measures) replace_all(s, p.first, p.second);
            } else if (rule.kind == "asmd") {
                while (apply(rule, s)) {}
            } else apply(rule, s);
        }
        for (const auto &p : post) replace_all(s, p.first, p.second);
        return filtered(s, after);
    }
    std::vector<Text> normalize(View input) const {
        Text remove = before + U' ';
        auto clean = filtered(input, remove);
        auto utf8 = encode(clean);
        auto separators = splitter->find(utf8);
        std::string separated;
        size_t offset = 0;
        for (const auto &match : separators) {
            separated.append(utf8, offset, match.end - offset);
            separated.push_back('\n');
            offset = match.end;
        }
        separated.append(utf8, offset, utf8.size() - offset);
        auto text = decode(separated);
        auto trimmed = trim(text);
        std::vector<Text> result;
        for (auto line : split(trimmed, U'\n')) {
            if (!line.empty() || result.empty()) result.push_back(sentence(trim(line)));
        }
        return result;
    }
};
TextNormalizer::TextNormalizer(const std::string &directory, const PypinyinResolver &dictionary)
    : impl_(std::make_unique<Impl>(directory, dictionary)) {}
TextNormalizer::~TextNormalizer() = default;
std::vector<std::u32string> TextNormalizer::normalize(std::u32string_view text) const {
    return impl_->normalize(text);
}
} // namespace tts::text
