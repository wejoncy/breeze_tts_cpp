// Chinese phonemization follows GPT-SoVITS (MIT, Copyright (c) 2024 RVC-Boss)
// and PaddleSpeech's pronunciation rules (Apache-2.0).
#include "text/phonemizer.h"
#include "text/normalization.h"
#include "text/weighted_segmenter.h"
#include "text/tone_sandhi.h"
#include "text/linguistic_data.h"
#include "text/utf8.h"
#include "io/json.h"
#include <algorithm>
#include <array>
#include <map>
#include <set>
namespace tts::text {
struct ChinesePhonemizer::Impl {
    WeightedSegmenter segmenter;
    PypinyinResolver dictionary;
    TextNormalizer normalizer;
    ToneSandhi sandhi;
    std::map<std::u32string, std::vector<std::string>, std::less<>> corrections;
    std::map<std::string, std::array<std::string, 2>, std::less<>> phones;
    std::map<std::string, int, std::less<>> symbol_ids;
    std::map<std::string, std::string> consonant_finals, zero_finals, zero_prefix;
    std::set<std::u32string, std::less<>> must_erhua, not_erhua;
    std::vector<std::pair<std::u32string, std::u32string>> replacements;
    std::u32string punctuation;
    std::map<char32_t, char32_t> aliases;
    std::vector<std::pair<uint32_t, uint32_t>> speakable_ranges;
    int unknown;
    Impl(const std::string &directory)
        : segmenter(directory), dictionary(directory), normalizer(directory, dictionary),
          sandhi(directory, segmenter, dictionary) {
        auto data = read_json(directory + "/linguistics.json");
        for (const auto &value : data.at("punctuation")) punctuation += decode(value.get<std::string>());
        speakable_ranges = data.at("speakable_ranges").get<decltype(speakable_ranges)>();
        for (auto it = data.at("aliases").begin(); it != data.at("aliases").end(); ++it) {
            auto source = decode(it.key()), target = decode(it.value().get<std::string>());
            if (source.size() != 1 || target.size() != 1) throw std::invalid_argument("phonemization alias must preserve codepoint length");
            aliases.emplace(source.front(), target.front());
        }
        for (const auto &pair : data.at("replacements"))
            replacements.emplace_back(decode(pair[0].get<std::string>()), decode(pair[1].get<std::string>()));
        for (auto item : {std::pair{"must_erhua", &must_erhua}, std::pair{"not_erhua", &not_erhua}})
            for (const auto &word : data.at(item.first)) item.second->insert(decode(word.get<std::string>()));
        auto &rules = data.at("phoneme_normalization");
        consonant_finals = rules.at("v_rep_map").get<decltype(consonant_finals)>();
        zero_finals = rules.at("pinyin_rep_map").get<decltype(zero_finals)>();
        zero_prefix = rules.at("single_rep_map").get<decltype(zero_prefix)>();
        for (auto it = data.at("phones").begin(); it != data.at("phones").end(); ++it)
            phones.emplace(it.key(), it.value().get<std::array<std::string, 2>>());
        int id = 0;
        for (const auto &value : data.at("symbols")) {
            if (!symbol_ids.emplace(value.get<std::string>(), id++).second)
                throw std::invalid_argument("duplicate pronunciation symbol");
        }
        unknown = symbol_ids.at("UNK");
        MappedFile file(directory + "/polyphone.bin");
        DataReader reader(file.data(), file.size());
        if (reader.text(8) != "GSVPOLY1") throw std::invalid_argument("unsupported pronunciation correction asset");
        for (int section = 0; section < 2; ++section) {
            auto count = reader.read<uint32_t>();
            if (count > reader.remaining() / 3) throw std::invalid_argument("pronunciation correction count");
            while (count--) {
                auto n = reader.read<uint8_t>();
                std::u32string word;
                while (n--) word.push_back(reader.read<uint32_t>());
                n = reader.read<uint8_t>();
                std::vector<std::string> values;
                while (n--) values.emplace_back(reader.text(reader.read<uint8_t>()));
                if (word.empty() || values.empty() || (section == 1 && word.size() != 1))
                    throw std::invalid_argument("invalid pronunciation correction");
                corrections.insert_or_assign(std::move(word), std::move(values));
            }
        }
        reader.finish();
    }
    bool punct(char32_t cp) const { return punctuation.find(cp) != std::u32string::npos; }
    std::u32string replace(std::u32string_view sentence) const {
        std::u32string out;
        for (size_t first = 0; first < sentence.size();) {
            auto found = std::find_if(replacements.begin(), replacements.end(), [&](const auto &pair) {
                return sentence.substr(first, pair.first.size()) == pair.first;
            });
            if (found != replacements.end()) { out += found->second; first += found->first.size(); }
            else {
                char32_t cp = sentence[first++];
                auto alias = aliases.find(cp);
                if (alias != aliases.end()) cp = alias->second;
                out.push_back(cp);
            }
        }
        out.erase(std::remove_if(out.begin(), out.end(), [&](char32_t cp) {
            return !punct(cp) && !std::any_of(speakable_ranges.begin(), speakable_ranges.end(),
                [&](auto range) { return cp >= range.first && cp <= range.second; });
        }), out.end());
        return out;
    }
    void correct(std::u32string_view word, std::vector<std::string> &readings) const {
        auto found = corrections.find(word);
        if (word.size() > 1 && found != corrections.end()) {
            if (found->second.size() == readings.size()) readings = found->second;
            return;
        }
        for (size_t i = 0; i < word.size() && i < readings.size(); ++i) {
            auto entry = corrections.find(word.substr(i, 1));
            if (entry != corrections.end()) readings[i] = entry->second.front();
        }
    }
    void erhua(std::u32string_view word, std::string_view pos, std::vector<std::string> &finals) const {
        if (!finals.empty() && finals.back() == "er1" && word.at(finals.size() - 1) == U'儿') finals.back() = "er2";
        if (must_erhua.find(word) == must_erhua.end() &&
            (not_erhua.find(word) != not_erhua.end() || pos == "a" || pos == "j" || pos == "nr")) return;
        if (finals.size() != word.size()) return;
        if (finals.size() >= 2 && word.back() == U'儿' &&
            (finals.back() == "er2" || finals.back() == "er5") &&
            not_erhua.find(word.substr(word.size() - 2)) == not_erhua.end()) {
            if (finals[finals.size() - 2].empty()) throw std::invalid_argument("empty erhua preceding final");
            finals.back().back() = finals[finals.size() - 2].back();
        }
    }
    void append(const std::string &initial, const std::string &final, Phonemes &out) const {
        auto symbol = [&](const std::string &phone) {
            auto it = symbol_ids.find(phone);
            out.phones.push_back(it != symbol_ids.end() ? it->second : unknown);
        };
        if (initial == final) {
            auto cp = decode(final);
            if (cp.size() != 1 || !punct(cp.front())) throw std::invalid_argument("non-punctuation raw pronunciation: " + final);
            symbol(final); out.word2ph.push_back(1); return;
        }
        if (final.empty() || final.back() < '1' || final.back() > '5') throw std::invalid_argument("invalid final tone: " + final);
        auto vowel = final.substr(0, final.size() - 1);
        std::string syllable = initial + vowel;
        if (!initial.empty()) {
            auto it = consonant_finals.find(vowel);
            if (it != consonant_finals.end()) syllable = initial + it->second;
        } else {
            auto it = zero_finals.find(syllable);
            if (it != zero_finals.end()) syllable = it->second;
            else if (!syllable.empty()) {
                auto first = zero_prefix.find(syllable.substr(0, 1));
                if (first != zero_prefix.end()) syllable = first->second + syllable.substr(1);
            }
        }
        auto found = phones.find(syllable);
        if (found == phones.end()) throw std::invalid_argument("phoneme mapping missing for " + syllable);
        symbol(found->second[0]); symbol(found->second[1] + final.back()); out.word2ph.push_back(2);
    }
};
ChinesePhonemizer::ChinesePhonemizer(const std::string &directory) : impl_(std::make_unique<Impl>(directory)) {}
ChinesePhonemizer::~ChinesePhonemizer() = default;
const PypinyinResolver &ChinesePhonemizer::dictionary() const { return impl_->dictionary; }
std::u32string ChinesePhonemizer::normalize(std::u32string_view input) const {
    std::u32string joined;
    for (const auto &sentence : impl_->normalizer.normalize(input)) joined += impl_->replace(sentence);
    auto end = std::unique(joined.begin(), joined.end(), [&](char32_t a, char32_t b) { return impl_->punct(a) && impl_->punct(b); });
    joined.erase(end, joined.end());
    return joined;
}
std::vector<std::u32string> ChinesePhonemizer::segments(std::u32string_view normalized) const {
    std::vector<std::u32string> result;
    size_t first = 0;
    for (size_t i = 0; i < normalized.size(); ++i) if (impl_->punct(normalized[i])) {
        result.emplace_back(normalized.substr(first, i + 1 - first)); first = i + 1;
    }
    if (first < normalized.size()) result.emplace_back(normalized.substr(first));
    return result;
}
Phonemes ChinesePhonemizer::pronounce(std::u32string_view normalized, const PinyinResolver &resolver) const {
    Phonemes result;
    for (const auto &sentence : segments(normalized)) {
        std::vector<std::string> readings;
        bool contextual = resolver.resolveSentenceTone3(sentence.data(), sentence.size(), &readings);
        if (contextual && readings.size() != sentence.size()) throw std::invalid_argument("contextual pronunciation alignment mismatch");
        auto tokens = impl_->sandhi.premerge(impl_->segmenter.segment(sentence));
        size_t cursor = 0;
        for (const auto &token : tokens) {
            std::vector<std::string> initials, finals;
            if (contextual) {
                if (cursor > sentence.size() || token.word.size() > sentence.size() - cursor ||
                    sentence.substr(cursor, token.word.size()) != token.word) throw std::invalid_argument("contextual word alignment mismatch");
                std::vector<std::string> word_readings(readings.begin() + cursor, readings.begin() + cursor + token.word.size());
                impl_->correct(token.word, word_readings);
                for (const auto &reading : word_readings) {
                    bool pinyin = !reading.empty() && ((reading.front() >= 'a' && reading.front() <= 'z') || (reading.front() >= 'A' && reading.front() <= 'Z'));
                    auto value = pinyin ? impl_->dictionary.split(reading) : std::pair{reading, reading};
                    initials.push_back(std::move(value.first)); finals.push_back(std::move(value.second));
                }
            } else {
                std::vector<PinyinSyl> syllables;
                resolver.resolve(token.word.data(), token.word.size(), &syllables);
                for (const auto &syllable : syllables) {
                    auto value = syllable.raw ? std::pair{syllable.text, syllable.text} : impl_->dictionary.split(syllable.text);
                    initials.push_back(std::move(value.first)); finals.push_back(std::move(value.second));
                }
            }
            cursor += token.word.size();
            impl_->sandhi.modify(token.word, token.flag, finals);
            impl_->erhua(token.word, token.flag, finals);
            if (initials.size() != finals.size()) throw std::invalid_argument("initial/final alignment mismatch");
            for (size_t i = 0; i < initials.size(); ++i) impl_->append(initials[i], finals[i], result);
        }
    }
    return result;
}
} // namespace tts::text
