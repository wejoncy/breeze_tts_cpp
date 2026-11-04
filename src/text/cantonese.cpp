// Pronunciation selection: ToJyutping (BSD-2-Clause). Phone mapping: GPT-SoVITS (MIT).
#include "text/cantonese.h"
#include "text/lexicon.h"
#include "text/normalization.h"
#include "text/utf8.h"
#include "io/json.h"
#include <algorithm>
#include <map>

namespace tts::text {
struct CantonesePhonemizer::Impl {
    Lexicon<int> pronunciations;
    TextNormalizer normalizer;
    std::vector<std::pair<std::u32string, std::u32string>> replacements;
    std::map<char32_t, int> punctuation;
    Impl(const std::string &directory, const PypinyinResolver &dictionary)
        : normalizer(directory, dictionary) {
        auto data = read_json(directory + "/linguistics.json");
        const auto symbols = data.at("symbols").get<std::vector<std::string>>();
        const auto &rules = data.at("cantonese");
        for (const auto &value : rules.at("replacements"))
            replacements.emplace_back(decode(value[0].get<std::string>()),
                                      decode(value[1].get<std::string>()));
        for (const auto &entry : replacements)
            if (entry.first.empty())
                throw std::invalid_argument("empty Cantonese replacement");
        for (const auto &value : data.at("punctuation")) {
            auto s = value.get<std::string>();
            auto cp = decode(s);
            auto id = std::find(symbols.begin(), symbols.end(), s);
            if (cp.size() != 1 || id == symbols.end())
                throw std::invalid_argument("invalid Cantonese punctuation asset");
            punctuation.emplace(cp.front(), int(id - symbols.begin()));
        }
        pronunciations = load_lexicon<int>(read_json(directory + "/cantonese.json"));
        for (const auto &[word, phones] : pronunciations)
            for (int phone : phones)
                if (phone < 0 || size_t(phone) >= symbols.size())
                    throw std::invalid_argument("Cantonese phone outside vocabulary");
    }
    std::u32string normalize(std::u32string_view text) const {
        std::u32string out;
        for (const auto &sentence : normalizer.normalize(text)) {
            for (size_t i = 0; i < sentence.size();) {
                auto found = std::find_if(replacements.begin(), replacements.end(),
                    [&](const auto &entry) {
                        return std::u32string_view(sentence).substr(i, entry.first.size()) == entry.first;
                    });
                std::u32string_view next;
                if (found != replacements.end()) {
                    next = found->second;
                    i += found->first.size();
                } else next = std::u32string_view(sentence).substr(i++, 1);
                for (auto cp : next)
                    if ((cp >= 0x4e00 && cp <= 0x9fa5) || punctuation.count(cp))
                        out += cp;
            }
        }
        return out;
    }
    std::vector<int> pronounce(std::u32string_view text) const {
        std::vector<int> out;
        for (size_t i = 0; i < text.size();) {
            if (auto found = punctuation.find(text[i]); found != punctuation.end()) {
                out.push_back(found->second);
                ++i;
                continue;
            }
            const auto selected = longest_match(pronunciations, text.substr(i));
            if (selected == pronunciations.end())
                throw std::invalid_argument("Cantonese dictionary has no pronunciation for " +
                                            encode(text.substr(i, 1)));
            const auto &phones = selected->second;
            if (phones.empty())
                throw std::invalid_argument("Cantonese pronunciation/character alignment mismatch");
            out.insert(out.end(), phones.begin(), phones.end());
            i += selected->first.size();
        }
        return out;
    }
};
CantonesePhonemizer::CantonesePhonemizer(const std::string &directory,
                                       const PypinyinResolver &dictionary)
    : impl_(std::make_unique<Impl>(directory, dictionary)) {}
CantonesePhonemizer::~CantonesePhonemizer() = default;
std::u32string CantonesePhonemizer::normalize(std::u32string_view text) const {
    return impl_->normalize(text);
}
std::vector<int> CantonesePhonemizer::pronounce(std::u32string_view text) const {
    return impl_->pronounce(text);
}
} // namespace tts::text
