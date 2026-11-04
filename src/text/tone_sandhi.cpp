// Tone sandhi rules follow PaddleSpeech (Apache-2.0, Copyright (c) 2021
// PaddlePaddle Authors), as used by GPT-SoVITS's Chinese frontend.
#include "text/tone_sandhi.h"
#include "text/pinyin.h"
#include "text/utf8.h"
#include "io/json.h"
#include <algorithm>
#include <map>
#include <set>
namespace tts::text {
namespace {
char tone(const std::string &final) {
    if (final.empty()) throw std::invalid_argument("empty pronunciation final");
    return final.back();
}
void set_tone(std::string &final, char value) { tone(final); final.back() = value; }
bool all_three(const std::vector<std::string> &finals, size_t first, size_t last) {
    for (size_t i = first; i < last; ++i) if (tone(finals.at(i)) != '3') return false;
    return true;
}
bool reduplicated(std::u32string_view word) { return word.size() == 2 && word.front() == word.back(); }
bool contains(std::u32string_view characters, char32_t cp) { return characters.find(cp) != std::u32string_view::npos; }
}
struct ToneSandhi::Impl {
    const JiebaSegmenter &segmenter;
    const PypinyinResolver &dictionary;
    std::set<std::u32string, std::less<>> neutral, stressed;
    std::u32string punctuation;
    std::map<std::string, std::u32string> groups;
    Impl(const std::string &directory, const JiebaSegmenter &s, const PypinyinResolver &d) : segmenter(s), dictionary(d) {
        auto data = read_json(directory + "/linguistics.json");
        for (auto item : {std::pair{"neutral_words", &neutral}, std::pair{"not_neutral_words", &stressed}})
            for (const auto &value : data.at(item.first)) item.second->insert(decode(value.get<std::string>()));
        punctuation = decode(data.at("sandhi_punctuation").get<std::string>());
        for (auto it = data.at("sandhi").begin(); it != data.at("sandhi").end(); ++it)
            groups.emplace(it.key(), decode(it.value().get<std::string>()));
    }
    bool has(std::string_view key, char32_t cp) const { return contains(groups.at(std::string(key)), cp); }
    bool conventional(std::u32string_view word) const {
        return neutral.find(word) != neutral.end() || neutral.find(word.substr(word.size() > 2 ? word.size() - 2 : 0)) != neutral.end();
    }
    size_t split(std::u32string_view word) const {
        auto words = segmenter.search(word);
        if (words.empty()) throw std::invalid_argument("empty sandhi segmentation");
        auto it = std::min_element(words.begin(), words.end(), [](const auto &a, const auto &b) { return a.size() < b.size(); });
        return word.find(*it) == 0 ? it->size() : word.size() - it->size();
    }
    std::vector<SegToken> merge_three(const std::vector<SegToken> &segments, bool boundary) const {
        std::vector<std::vector<std::string>> finals;
        for (const auto &segment : segments) {
            std::vector<std::string> first, last;
            dictionary.initialsFinals(segment.word, first, last);
            if (last.empty() || std::any_of(last.begin(), last.end(), [](const auto &value) { return value.empty(); })) return segments;
            finals.push_back(std::move(last));
        }
        std::vector<bool> merged(segments.size(), false);
        std::vector<SegToken> result;
        for (size_t i = 0; i < segments.size(); ++i) {
            bool compatible = i && !merged[i - 1] &&
                (boundary ? tone(finals.at(i - 1).back()) == '3' && tone(finals.at(i).front()) == '3' :
                 all_three(finals.at(i - 1), 0, finals.at(i - 1).size()) && all_three(finals.at(i), 0, finals.at(i).size()));
            if (compatible && !reduplicated(segments[i - 1].word) && segments[i - 1].word.size() + segments[i].word.size() <= 3) {
                result.back().word += segments[i].word; merged[i] = true;
            } else result.push_back(segments[i]);
        }
        return result;
    }
    void neutralize(std::u32string_view word, std::string_view pos, std::vector<std::string> &finals) const {
        bool exception = stressed.find(word) != stressed.end();
        for (size_t i = 1; i < word.size(); ++i)
            if (word[i] == word[i - 1] && !pos.empty() && contains(U"nva", char32_t(pos.front())) && !exception)
                set_tone(finals.at(i), '5');
        auto ge = word.find(U'个');
        if (has("particles", word.back()) || has("structural", word.back()) ||
            (word.size() == 1 && has("aspect", word.back()) && (pos == "ul" || pos == "uz" || pos == "ug")) ||
            (word.size() > 1 && has("plural", word.back()) && (pos == "r" || pos == "n") && !exception) ||
            (word.size() > 1 && has("location", word.back()) && (pos == "s" || pos == "l" || pos == "f")) ||
            (word.size() > 1 && has("direction", word.back()) && has("direction_prefix", word[word.size() - 2])))
            set_tone(finals.back(), '5');
        else if ((ge != std::u32string_view::npos && ge >= 1 && (dictionary.isNumeric(word[ge - 1]) || has("ge_prefix", word[ge - 1]))) || word == U"个")
            set_tone(finals.at(ge), '5');
        else if (conventional(word)) set_tone(finals.back(), '5');
        auto first = split(word);
        for (auto range : {std::pair{size_t(0), first}, std::pair{first, word.size()}})
            if (range.first != range.second && conventional(word.substr(range.first, range.second - range.first)))
                set_tone(finals.at(range.second - 1), '5');
    }
    void third_tone(std::u32string_view word, std::vector<std::string> &finals) const {
        if (word.size() == 2 && all_three(finals, 0, finals.size())) set_tone(finals.at(0), '2');
        else if (word.size() == 3) {
            auto first = split(word);
            if (all_three(finals, 0, finals.size())) {
                if (first == 2) { set_tone(finals.at(0), '2'); set_tone(finals.at(1), '2'); }
                else if (first == 1) set_tone(finals.at(1), '2');
            } else {
                if (first == 2 && all_three(finals, 0, first)) set_tone(finals.at(0), '2');
                bool last_three = all_three(finals, first, finals.size());
                if (finals.size() - first == 2 && last_three) set_tone(finals.at(first), '2');
                else if (first && first < finals.size() && !last_three && tone(finals.at(first)) == '3' && tone(finals.at(first - 1)) == '3')
                    set_tone(finals.at(first - 1), '2');
            }
        } else if (word.size() == 4) {
            for (size_t first : {size_t(0), size_t(2)})
                if (all_three(finals, first, first + 2)) set_tone(finals.at(first), '2');
        }
    }
};
ToneSandhi::ToneSandhi(const std::string &directory, const JiebaSegmenter &segmenter, const PypinyinResolver &dictionary)
    : impl_(std::make_unique<Impl>(directory, segmenter, dictionary)) {}
ToneSandhi::~ToneSandhi() = default;
std::vector<SegToken> ToneSandhi::premerge(std::vector<SegToken> segments) const {
    std::vector<SegToken> result;
    std::u32string last;
    for (auto token : segments) {
        if (last == U"不") token.word = last + token.word;
        if (token.word != U"不") result.push_back(token);
        last = token.word;
    }
    if (last == U"不") result.push_back({last, "d"});
    segments = std::move(result); result.clear();
    for (size_t i = 0; i < segments.size(); ++i) {
        if (i && segments[i].word == U"一" && i + 1 < segments.size() && !result.empty() &&
            result.back().word == segments[i + 1].word && result.back().flag == "v" && segments[i + 1].flag == "v") {
            result.back().word += U"一" + segments[i + 1].word; ++i;
        } else result.push_back(segments[i]);
    }
    segments = std::move(result); result.clear();
    for (auto token : segments)
        if (!result.empty() && result.back().word == U"一") result.back().word += token.word;
        else result.push_back(std::move(token));
    segments = std::move(result); result.clear();
    for (auto token : segments)
        if (!result.empty() && token.word == result.back().word) result.back().word += token.word;
        else result.push_back(std::move(token));
    segments = impl_->merge_three(result, false);
    segments = impl_->merge_three(segments, true); result.clear();
    for (size_t i = 0; i < segments.size(); ++i)
        if (i && segments[i].word == U"儿" && segments[i - 1].word != U"#") result.back().word += segments[i].word;
        else result.push_back(segments[i]);
    return result;
}
void ToneSandhi::modify(std::u32string_view word, std::string_view pos, std::vector<std::string> &finals) const {
    if (word.empty() || finals.empty()) throw std::invalid_argument("sandhi word/finals alignment mismatch");
    if (word.size() == 3 && word[1] == U'不') set_tone(finals.at(1), '5');
    else for (size_t i = 0; i + 1 < word.size(); ++i)
        if (word[i] == U'不' && tone(finals.at(i + 1)) == '4') set_tone(finals.at(i), '2');
    bool numeric_yi = word.find(U'一') != std::u32string_view::npos &&
        std::all_of(word.begin(), word.end(), [&](char32_t cp) { return cp == U'一' || impl_->dictionary.isNumeric(cp); });
    if (!numeric_yi) {
        if (word.size() == 3 && word[1] == U'一' && word.front() == word.back()) set_tone(finals.at(1), '5');
        else if (word.size() >= 2 && word.substr(0, 2) == U"第一") set_tone(finals.at(1), '1');
        else for (size_t i = 0; i + 1 < word.size(); ++i) if (word[i] == U'一') {
            if (tone(finals.at(i + 1)) == '4') set_tone(finals.at(i), '2');
            else if (!contains(impl_->punctuation, word[i + 1])) set_tone(finals.at(i), '4');
        }
    }
    impl_->neutralize(word, pos, finals);
    impl_->third_tone(word, finals);
}
} // namespace tts::text
