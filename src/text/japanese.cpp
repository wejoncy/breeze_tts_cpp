// GPT-SoVITS Japanese normalization/prosody (MIT) over native Open JTalk rules.
#include "text/japanese.h"
#include "text/japanese_grammar.h"
#include "text/morphology.h"
#include "text/regex.h"
#include "text/utf8.h"
#include <algorithm>
#include <map>
#include <cstring>

namespace tts::text {
struct JapanesePhonemizer::Impl {
    japanese::Rules rules;
    Morphology morphology;
    Regex marks;
    std::map<char32_t, std::u32string> lower, symbol_replacements;
    std::map<std::string, std::string> replacements;
    std::map<std::string, int> symbols;
    std::u32string punctuation;
    explicit Impl(const std::string &directory)
        : rules(directory + "/japanese.json"), morphology(directory + "/japanese_morphology.bin"),
          marks(rules.data.at("marks").get<std::string>()) {
        auto data = read_json(directory + "/linguistics.json");
        int index = 0;
        for (const auto &symbol : data.at("symbols")) symbols.emplace(symbol.get<std::string>(), index++);
        if (!symbols.count("UNK")) throw std::invalid_argument("Japanese requires an unknown phoneme symbol");
        for (const auto &symbol : rules.data.at("punctuation")) punctuation += decode(symbol.get<std::string>());
        replacements = rules.data.at("replacements").get<decltype(replacements)>();
        for (const auto &entry : rules.data.at("symbol_replacements")) {
            auto from = decode(entry.at(0).get<std::string>());
            if (from.size() != 1) throw std::invalid_argument("Japanese symbol key");
            symbol_replacements.emplace(from.front(), decode(entry.at(1).get<std::string>()));
        }
        for (const auto &[from, to] : rules.data.at("lower").items()) {
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
        const auto &conversions = rules.table("text2mecab_conv_list");
        std::string output;
        for (size_t cursor = 0; cursor < input.size();) {
            size_t i = 0;
            for (; conversions.at(i); i += 2) {
                auto size = std::strlen(conversions[i]);
                if (std::string_view(input).substr(cursor, size) != conversions[i]) continue;
                output += conversions[i + 1]; cursor += size; break;
            }
            if (conversions.at(i)) continue;
            auto cp = static_cast<unsigned char>(input[cursor]);
            if (cp <= 0x1f) { ++cursor; continue; }
            auto size = cp < 0x80 ? 1u : cp < 0xe0 ? 2u : cp < 0xf0 ? 3u : 4u;
            output.append(input, cursor, size); cursor += size;
        }
        return output;
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
            japanese::Document document;
            for (const auto &token : morphology.analyze(mecab_input(std::string(text))))
                document.append(token.surface + "," + token.feature);
            japanese::pronounce(document, rules);
            auto result = japanese::prosody(document, rules);
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
