// GPT-SoVITS Japanese normalization/prosody (MIT) over native Open JTalk rules.
#include "text/japanese.h"
#include "io/json.h"
#include "text/morphology.h"
#include "text/regex.h"
#include "text/utf8.h"
#include <algorithm>
#include <map>
#include <charconv>
#include <climits>
#include <jpcommon.h>
#include <njd.h>
#include <mecab2njd.h>
#include <njd2jpcommon.h>
#include <njd_set_pronunciation.h>
#include <njd_set_digit.h>
#include <njd_set_accent_phrase.h>
#include <njd_set_accent_type.h>
#include <njd_set_unvoiced_vowel.h>
#include <text2mecab.h>

namespace tts::text {
namespace {
struct Labels {
    NJD words;
    JPCommon common;
    Labels() { NJD_initialize(&words); JPCommon_initialize(&common); }
    ~Labels() { JPCommon_clear(&common); NJD_clear(&words); }
};
int number(const std::string &label, std::string_view marker, int field = 0) {
    auto cursor = label.find(marker);
    if (cursor == label.npos) return -1;
    cursor += marker.size();
    for (int index = 0; index < field; ++index) {
        cursor = label.find('+', cursor);
        if (cursor == label.npos) return -1;
        ++cursor;
    }
    int value = -1;
    auto result = std::from_chars(label.data() + cursor, label.data() + label.size(), value);
    return result.ec == std::errc{} ? value : -1;
}
std::vector<std::string> prosody(const std::vector<Morphology::Token> &tokens) {
    if (tokens.empty()) return {};
    Labels labels;
    std::vector<std::string> features;
    for (const auto &token : tokens) features.push_back(token.surface + "," + token.feature);
    std::vector<char *> pointers;
    for (auto &feature : features) pointers.push_back(feature.data());
    mecab2njd(&labels.words, pointers.data(), int(pointers.size()));
    njd_set_pronunciation(&labels.words);
    njd_set_digit(&labels.words);
    njd_set_accent_phrase(&labels.words);
    njd_set_accent_type(&labels.words);
    njd_set_unvoiced_vowel(&labels.words);
    njd2jpcommon(&labels.common, &labels.words);
    JPCommon_make_label(&labels.common);
    const int count = JPCommon_get_label_size(&labels.common);
    auto **contexts = JPCommon_get_label_feature(&labels.common);
    std::vector<std::string> output;
    for (int index = 0; index < count; ++index) {
        const std::string label = contexts[index];
        const auto first = label.find('-'), last = label.find('+', first + 1);
        if (first == label.npos || last == label.npos) throw std::runtime_error("invalid Open JTalk phone label");
        auto phone = label.substr(first + 1, last - first - 1);
        if (phone == "sil") continue;
        if (phone == "pau") { output.emplace_back("_"); continue; }
        if (phone.size() == 1 && std::string_view("AEIOU").find(phone.front()) != std::string_view::npos)
            phone.front() += 'a' - 'A';
        output.push_back(phone);
        const int accent = number(label, "/A:"), position = number(label, "/A:", 1);
        const int remaining = number(label, "/A:", 2), length = number(label, "/F:");
        const int next = index + 1 < count ? number(contexts[index + 1], "/A:", 1) : -1;
        if (remaining == 1 && next == 1 && std::string_view("aeiouAEIOUNcl").find(phone) != std::string_view::npos)
            output.emplace_back("#");
        else if (accent == 0 && next == position + 1 && position != length) output.emplace_back("]");
        else if (position == 1 && next == 2) output.emplace_back("[");
    }
    return output;
}
} // namespace
struct JapanesePhonemizer::Impl {
    const Json rules;
    Morphology morphology;
    Regex marks;
    std::map<char32_t, std::u32string> lower, symbol_replacements;
    std::map<std::string, std::string> replacements;
    std::map<std::string, int> symbols;
    std::u32string punctuation;
    explicit Impl(const std::string &directory)
        : rules(read_json(directory + "/japanese.json")),
          morphology(directory + "/japanese_mecab", directory + "/japanese_user.dic"),
          marks(rules.at("marks").get<std::string>()) {
        auto data = read_json(directory + "/linguistics.json");
        int index = 0;
        for (const auto &symbol : data.at("symbols")) symbols.emplace(symbol.get<std::string>(), index++);
        if (!symbols.count("UNK")) throw std::invalid_argument("Japanese requires an unknown phoneme symbol");
        for (const auto &symbol : rules.at("punctuation")) punctuation += decode(symbol.get<std::string>());
        replacements = rules.at("replacements").get<decltype(replacements)>();
        for (const auto &entry : rules.at("symbol_replacements")) {
            auto from = decode(entry.at(0).get<std::string>());
            if (from.size() != 1) throw std::invalid_argument("Japanese symbol key");
            symbol_replacements.emplace(from.front(), decode(entry.at(1).get<std::string>()));
        }
        for (const auto &[from, to] : rules.at("lower").items()) {
            auto decoded = decode(from);
            if (decoded.size() != 1) throw std::invalid_argument("Japanese lowercase key");
            lower.emplace(decoded.front(), decode(to.get<std::string>()));
        }
    }
    std::u32string normalize(std::u32string_view input) const {
        std::u32string output;
        bool mark = false;
        for (auto cp : input) {
            bool selected = punctuation.find(cp) != punctuation.npos;
            if (!(selected && mark)) output += cp;
            mark = selected;
        }
        return output;
    }
    std::string mecab_input(const std::string &input) const {
        // Open JTalk 1.11's UTF-8 table expands ASCII to at most three bytes.
        if (input.size() > size_t(INT_MAX) / 3) throw std::length_error("Japanese text is too large");
        std::vector<char> converted(input.size() * 3 + 1);
        text2mecab(converted.data(), input.c_str());
        return converted.data();
    }
    std::vector<int> pronounce(std::u32string_view normalized) const {
        std::u32string prepared;
        for (auto cp : normalized) {
            auto symbol = symbol_replacements.find(cp);
            if (symbol != symbol_replacements.end()) { prepared += symbol->second; continue; }
            auto found = lower.find(cp);
            if (found != lower.end()) prepared += found->second;
            else prepared += cp;
        }
        auto input = encode(prepared);
        std::vector<std::string> phones;
        auto sentence = [&](std::string_view text) {
            if (text.empty()) return;
            auto result = prosody(morphology.analyze(mecab_input(std::string(text))));
            phones.insert(phones.end(), result.begin(), result.end());
        };
        size_t cursor = 0;
        for (auto mark : marks.find(input)) {
            sentence(std::string_view(input).substr(cursor, mark.start - cursor));
            auto value = input.substr(mark.start, mark.end - mark.start);
            if (value != " ") phones.push_back(std::move(value));
            cursor = mark.end;
        }
        sentence(std::string_view(input).substr(cursor));
        std::vector<int> result;
        for (auto phone : phones) {
            auto replacement = replacements.find(phone);
            if (replacement != replacements.end()) phone = replacement->second;
            auto found = symbols.find(phone);
            result.push_back(found == symbols.end() ? symbols.at("UNK") : found->second);
        }
        return result;
    }
};
JapanesePhonemizer::JapanesePhonemizer(const std::string &directory)
    : impl_(std::make_unique<Impl>(directory)) {}
JapanesePhonemizer::~JapanesePhonemizer() = default;
std::u32string JapanesePhonemizer::normalize(std::u32string_view text) const { return impl_->normalize(text); }
std::vector<int> JapanesePhonemizer::pronounce(std::u32string_view text) const { return impl_->pronounce(text); }
} // namespace tts::text
