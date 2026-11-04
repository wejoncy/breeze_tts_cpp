// GPT-SoVITS LangSegmenter (MIT) and split-lang (MIT), native policy adaptation.
#include "text/language_router.h"
#include "text/budoux.h"
#include "text/language_detector.h"
#include "text/regex.h"
#include "text/utf8.h"
#include "io/json.h"
#include <algorithm>
#include <map>

namespace tts::text {
namespace {
using Text = std::u32string;
using Part = LanguageRouter::Span;
using Parts = std::vector<Part>;
using Ranges = std::vector<std::pair<char32_t, char32_t>>;
Ranges ranges(const Json &data) {
    Ranges result;
    for (const auto &entry : data) result.emplace_back(entry.at(0), entry.at(1));
    std::sort(result.begin(), result.end());
    return result;
}
bool contains(const Ranges &ranges, char32_t cp) {
    auto found = std::upper_bound(ranges.begin(), ranges.end(), cp,
        [](char32_t value, auto range) { return value < range.first; });
    return found != ranges.begin() && cp <= (--found)->second;
}
bool kana(char32_t cp) { return (cp >= 0x3040 && cp <= 0x30ff) || cp == U'々'; }
bool kana(std::u32string_view text) { return std::any_of(text.begin(), text.end(), [](auto c) { return kana(c); }); }
bool hangul(char32_t cp) { return cp >= 0xac00 && cp <= 0xd7af; }
bool cjk(char32_t cp) { return (cp >= 0x4e00 && cp <= 0x9fff) || kana(cp); }
void append(Parts &out, Part part) {
    if (!out.empty() && out.back().language == part.language) out.back().text += part.text;
    else out.push_back(std::move(part));
}
void merge(Parts &parts) {
    Parts out;
    for (auto &part : parts) append(out, std::move(part));
    parts.swap(out);
}
enum class Script { Other, ChineseJapanese, Korean, Digit, Punctuation, Newline };
struct Section { Script script; Parts parts; };
std::string nearby(const Parts &parts, size_t index, bool left) {
    for (size_t step = 1; step < parts.size(); ++step) {
        if ((left && step > index) || (!left && step >= parts.size() - index)) break;
        const auto &part = parts[left ? index - step : index + step];
        if (part.language != "x" && part.language != "digit") return part.language;
    }
    return parts[index].language;
}
} // namespace
struct LanguageRouter::Impl {
    text::LanguageDetector detector;
    text::Budoux japanese, chinese;
    text::Regex japanese_parts, korean_parts, cjk_punctuation;
    Ranges space, digits, cjk_ranges;
    Text punctuation;
    std::map<std::string, std::string> language_map, chinese_japanese_map;
    explicit Impl(const std::string &directory)
        : Impl(directory, read_json(directory + "/language_splitter.json")) {}
    Impl(const std::string &directory, const Json &data)
        : detector(directory), japanese(data.at("budoux").at("ja")),
          chinese(data.at("budoux").at("zh")),
          japanese_parts(data.at("japanese_parts").get<std::string>()),
          korean_parts(data.at("korean_parts").get<std::string>()),
          cjk_punctuation(data.at("cjk_punctuation").get<std::string>()),
          space(ranges(data.at("space"))), digits(ranges(data.at("digit"))),
          cjk_ranges(ranges(data.at("cjk_ranges"))),
          punctuation(text::decode(data.at("punctuation").get<std::string>())),
          language_map(data.at("language_map").get<decltype(language_map)>()),
          chinese_japanese_map(data.at("chinese_japanese_map").get<decltype(chinese_japanese_map)>()) {}
    Text strip(std::u32string_view value) const {
        size_t first = 0, end = value.size();
        while (first < end && contains(space, value[first])) ++first;
        while (end > first && contains(space, value[end - 1])) --end;
        return Text(value.substr(first, end - first));
    }
    bool full_english(std::u32string_view value) const {
        bool letters = false, first_line = true;
        for (auto cp : value) {
            if (cp == U'\n') first_line = false;
            bool latin = (cp >= U'A' && cp <= U'Z') || (cp >= U'a' && cp <= U'z');
            letters |= latin && first_line;
            if (!(latin || contains(space, cp) || (cp >= 0x20 && cp <= 0x7e) ||
                  (cp >= 0x2000 && cp <= 0x206f) || (cp >= 0x3000 && cp <= 0x303f) ||
                  (cp >= 0xff00 && cp <= 0xffef))) return false;
        }
        return letters;
    }
    Text full_cjk(std::u32string_view value) const {
        Text result;
        for (auto cp : value) {
            if (contains(cjk_ranges, cp) || !cjk_punctuation.find(text::encode(Text(1, cp))).empty()) result += cp;
        }
        return result;
    }
    struct Pass {
        const Impl &owner;
        std::map<Text, std::vector<std::string>> possible_cache;
        explicit Pass(const Impl &value) : owner(value) {}
        std::string classify(const Text &text, Script script) const {
            if (script == Script::ChineseJapanese && kana(text)) return "ja";
            auto prediction = owner.detector.detect(text);
            auto language = prediction.front().language;
            const auto &map = script == Script::ChineseJapanese ? owner.chinese_japanese_map : owner.language_map;
            auto found = map.find(language);
            return found == map.end() ? "x" : found->second;
        }
        const std::vector<std::string> &possible(Text text) {
            text.erase(std::remove(text.begin(), text.end(), U'\n'), text.end());
            text = owner.strip(text);
            auto [entry, inserted] = possible_cache.try_emplace(std::move(text));
            if (inserted)
                for (const auto &candidate : owner.detector.detect(entry->first, 5, 0.01f))
                    entry->second.push_back(candidate.language);
            return entry->second;
        }
        bool possible(const Part &part, const std::string &language) {
            const auto &languages = possible(part.text);
            return std::find(languages.begin(), languages.end(), language) != languages.end();
        }
        void middle(Parts &parts) {
            for (size_t i = 0; i + 2 < parts.size(); ++i) {
                auto &left = parts[i], &current = parts[i + 1], &right = parts[i + 2];
                if (left.language != right.language || left.language == "x" || left.language == "newline") continue;
                const bool no_kana = !kana(current.text);
                if (current.language == "x" || (no_kana && current.text.size() <= 4 && left.text.size() + right.text.size() >= 6) ||
                    (no_kana && possible(current, left.language))) current.language = left.language;
            }
        }
        bool short_near(const Part &current, const Part &near) {
            if (kana(current.text) || kana(near.text)) return false;
            return (near.language == "zh" && near.text.size() > current.text.size() && near.text.size() >= 6) || possible(current, near.language);
        }
        void sides(Parts &parts) {
            auto join = [&](size_t current, size_t near, bool left) {
                bool needs = parts[current].language == "x";
                if (parts.size() > 1) needs |= short_near(parts[current], parts[near]) ||
                    (parts[near].text.size() <= 5 && possible(parts[current], parts[near].language));
                if (needs) parts[current].language = nearby(parts, current, left);
            };
            join(0, std::min<size_t>(1, parts.size() - 1), false);
            join(parts.size() - 1, parts.size() > 1 ? parts.size() - 2 : 0, true);
        }
        void smart(Parts &parts) {
            if (parts.empty()) return;
            merge(parts); middle(parts); merge(parts);
            // Upstream's _get_languages assigns only a local variable; it does
            // not change parts. Avoid repeating that unused classifier work.
            middle(parts); merge(parts);
            for (size_t i = 0; i < parts.size(); ++i) if (parts[i].language == "x") {
                bool left = i + 1 == parts.size() || (i && parts[i - 1].text.size() >= parts[i + 1].text.size());
                parts[i].language = nearby(parts, i, left);
            }
            sides(parts); merge(parts); middle(parts); merge(parts);
        }
    };
    std::vector<Text> split_cjk(const Text &value) const {
        std::vector<Text> words;
        for (const auto &chunk : japanese.split(value))
            for (auto part : chinese.split(chunk)) {
                bool same = !words.empty() && kana(words.back()) == kana(part);
                if (same && (kana(part) || part.size() == 1)) words.back() += part;
                else words.push_back(std::move(part));
            }
        if (words.size() > 1 && words.front().size() == 1) {
            words[1] = words[0] + words[1]; words.erase(words.begin());
        }
        return words;
    }
    std::vector<Text> split_other(const Text &value) const {
        std::vector<Text> out;
        Text current;
        bool blanks = false;
        for (auto cp : value) {
            if (!contains(space, cp)) {
                if (blanks) { out.push_back(std::move(current)); current.clear(); blanks = false; }
            } else blanks = true;
            current += cp;
        }
        if (!current.empty()) out.push_back(std::move(current));
        return out;
    }
    std::vector<Section> sections(const Text &value, Pass &pass) const {
        struct Raw { Script script; Text text; };
        std::vector<Raw> raw;
        Script script = Script::Other;
        Text current;
        auto flush = [&] { if (!current.empty()) { raw.push_back({script, std::move(current)}); current.clear(); } };
        for (size_t i = 0; i < value.size(); ++i) {
            auto cp = value[i];
            Script selected = script;
            if (cjk(cp)) selected = Script::ChineseJapanese;
            else if (hangul(cp)) selected = Script::Korean;
            else if (contains(digits, cp)) selected = Script::Digit;
            else if (punctuation.find(cp) != Text::npos) {
                if (cp != U'\'' || !i || contains(space, value[i - 1])) { flush(); selected = Script::Punctuation; }
            } else if (contains(space, cp)) {
                if (cp == U'\n' || cp == U'\r') { flush(); selected = Script::Newline; }
            } else selected = Script::Other;
            if (selected != script) { flush(); script = selected; }
            current += cp;
        }
        flush();
        std::vector<Section> out;
        for (auto &item : raw) {
            Parts parts;
            switch (item.script) {
            case Script::Punctuation: parts.push_back({"punctuation", std::move(item.text)}); break;
            case Script::Digit: parts.push_back({"digit", std::move(item.text)}); break;
            case Script::Newline: parts.push_back({"newline", std::move(item.text)}); break;
            case Script::Korean: parts.push_back({"ko", std::move(item.text)}); break;
            default:
                for (auto word : item.script == Script::ChineseJapanese ? split_cjk(item.text) : split_other(item.text)) {
                    auto language = pass.classify(word, item.script);
                    parts.push_back({std::move(language), std::move(word)});
                }
            }
            if (item.script != Script::Punctuation && item.script != Script::Newline) pass.smart(parts);
            out.push_back({item.script, std::move(parts)});
        }
        return out;
    }
    static void across(std::vector<Section> &sections, Script selected) {
        if (sections.empty()) return;
        std::vector<Section> out;
        for (auto &section : sections) {
            if (out.empty()) { out.push_back(std::move(section)); continue; }
            auto &previous = out.back();
            if (previous.script == selected) {
                previous.script = section.script;
                if (selected == Script::Punctuation) {
                    previous.parts.back().text += section.parts.front().text;
                    previous.parts.back().language = section.parts.front().language;
                    previous.parts.insert(previous.parts.end(), std::make_move_iterator(section.parts.begin() + 1), std::make_move_iterator(section.parts.end()));
                } else previous.parts.insert(previous.parts.end(), std::make_move_iterator(section.parts.begin()), std::make_move_iterator(section.parts.end()));
            } else if (section.script == selected) {
                if (selected == Script::Punctuation) {
                    previous.parts.back().text += section.parts.front().text;
                    previous.parts.insert(previous.parts.end(), std::make_move_iterator(section.parts.begin() + 1), std::make_move_iterator(section.parts.end()));
                } else previous.parts.insert(previous.parts.end(), std::make_move_iterator(section.parts.begin()), std::make_move_iterator(section.parts.end()));
            } else out.push_back(std::move(section));
        }
        sections.clear();
        for (auto &section : out) {
            if (!sections.empty() && sections.back().script == section.script)
                sections.back().parts.insert(sections.back().parts.end(), std::make_move_iterator(section.parts.begin()), std::make_move_iterator(section.parts.end()));
            else sections.push_back(std::move(section));
        }
        const std::string name = selected == Script::Punctuation ? "punctuation" : "newline";
        for (auto &section : sections) {
            Parts merged;
            std::string last;
            for (auto &part : section.parts) {
                auto language = part.language;
                if (!merged.empty() && language == name) merged.back().text += part.text;
                else if (!merged.empty() && (language == last || last.empty())) {
                    if (merged.back().language == name) merged.back().language = language;
                    merged.back().text += part.text;
                } else merged.push_back(std::move(part));
                last = std::move(language);
            }
            section.parts.swap(merged);
        }
    }
    static void special_cjk(Parts &parts) {
        if (parts.size() < 2) return;
        const std::vector<std::string> order{"zh", "ja", "x", "digit", "punctuation", "newline"};
        std::map<std::string, size_t> lengths;
        for (const auto &name : order) lengths.emplace(name, 0);
        auto supported = [](const std::string &name) { return name == "zh" || name == "ja"; };
        Parts out;
        for (size_t i = 0; i < parts.size(); ++i) {
            auto &current = parts[i];
            lengths.at(current.language) += current.text.size();
            if (!i && supported(parts[i + 1].language) && (supported(current.language) || current.language == "x") &&
                current.text.size() * 10 < parts[i + 1].text.size()) {
                out.push_back({parts[i + 1].language, current.text + parts[i + 1].text}); ++i;
            } else if (i + 1 == parts.size() && supported(out.back().language) && (supported(current.language) || current.language == "x") &&
                       current.text.size() * 10 < out.back().text.size()) out.back().text += current.text;
            else if (i && i + 1 < parts.size() && out.back().language == parts[i + 1].language && supported(out.back().language) && current.language != "en" &&
                     current.text.size() * 10 < out.back().text.size() + parts[i + 1].text.size()) {
                out.back().text += current.text + parts[i + 1].text; ++i;
            } else out.push_back(std::move(current));
        }
        std::string most = order.front();
        for (const auto &name : order) if (lengths[name] > lengths[most]) most = name;
        for (auto &part : out) {
            if (lengths["x"] && part.language == "x") part.language = most;
            if (lengths["ja"] >= lengths["zh"] * 10 && part.language == "zh") part.language = "ja";
        }
        merge(out); parts.swap(out);
    }
    Parts split_script(const Part &part, const text::Regex &pattern, const std::string &language) const {
        Parts result;
        auto encoded = text::encode(part.text);
        size_t cursor = 0;
        for (auto match : pattern.find(encoded)) {
            if (match.start > cursor) result.push_back({part.language, text::decode(std::string_view(encoded).substr(cursor, match.start - cursor))});
            result.push_back({language, text::decode(std::string_view(encoded).substr(match.start, match.end - match.start))});
            cursor = match.end;
        }
        if (cursor < encoded.size()) result.push_back({part.language, text::decode(std::string_view(encoded).substr(cursor))});
        return result;
    }
    Parts segment(const Text &input, const std::string &default_language) const {
        Pass pass(*this);
        auto blocks = sections(strip(input), pass);
        across(blocks, Script::Punctuation); across(blocks, Script::Newline);
        Parts raw;
        for (auto &section : blocks) {
            if (section.script == Script::ChineseJapanese) special_cjk(section.parts);
            raw.insert(raw.end(), std::make_move_iterator(section.parts.begin()), std::make_move_iterator(section.parts.end()));
        }
        Parts result;
        bool numbers = false;
        auto filtered = [&](Part part) {
            if (part.language == "x") {
                auto value = full_cjk(part.text);
                if (!value.empty()) part = {"zh", std::move(value)};
            }
            append(result, std::move(part));
        };
        for (auto &part : raw) {
            if (part.language == "digit") {
                if (!default_language.empty()) part.language = default_language;
                else numbers = true;
                append(result, std::move(part)); continue;
            }
            if (full_english(part.text)) { part.language = "en"; append(result, std::move(part)); continue; }
            if (!default_language.empty()) { part.language = default_language; append(result, std::move(part)); continue; }
            auto ja = part.language == "ja" ? Parts{} : split_script(part, japanese_parts, "ja");
            if (ja.empty()) ja.push_back(part);
            Parts separated, ko;
            for (const auto &item : ja) {
                // Preserve upstream's ko_list lifetime across this loop.
                if (item.language != "ko") ko = split_script(item, korean_parts, "ko");
                if (!ko.empty()) separated.insert(separated.end(), ko.begin(), ko.end());
                else separated.push_back(item);
            }
            if (separated.size() == 1) filtered(part);
            else for (auto &item : separated) filtered(std::move(item));
        }
        if (numbers) {
            Parts joined;
            const Text marks = U",.!?，。！？";
            for (size_t i = 0; i < result.size(); ++i) {
                auto part = std::move(result[i]);
                if (part.language == "digit") {
                    if (!default_language.empty()) part.language = default_language;
                    else if (!joined.empty() && i + 1 == result.size()) part.language = joined.back().language;
                    else if (joined.empty() && i + 1 < result.size()) part.language = result[1].language;
                    else if (!joined.empty() && i + 1 < result.size()) {
                        const auto &left = joined.back(), &right = result[i + 1];
                        bool use_left = left.language == right.language ||
                            marks.find(right.text.front()) != Text::npos ||
                            part.text.back() == U'。' || part.text.back() == U'.' || left.text.size() >= right.text.size();
                        if (left.language != right.language && marks.find(left.text.back()) != Text::npos) use_left = false;
                        part.language = use_left ? left.language : right.language;
                    } else part.language = "zh";
                }
                append(joined, std::move(part));
            }
            result.swap(joined);
        }
        Parts final;
        for (auto &part : result) {
            if (part.language == "x") part.language = !final.empty() ? final.back().language : result.size() > 1 ? result[1].language : "zh";
            append(final, std::move(part));
        }
        return final;
    }
    Parts split(std::u32string_view value, std::string mode) const {
        Text input(value);
        input.erase(std::unique(input.begin(), input.end(), [](auto a, auto b) { return a == U' ' && b == U' '; }), input.end());
        if (mode.empty()) mode = "zh";
        if (mode == "en") return {{"en", std::move(input)}};
        const bool all = mode.compare(0, 4, "all_") == 0;
        auto parts = segment(input, all ? mode.substr(4) == "yue" ? "zh" : mode.substr(4) : "");
        if (mode == "auto" || mode == "auto_yue" || all) {
            if (mode == "auto_yue" || mode == "all_yue") for (auto &part : parts) if (part.language == "zh") part.language = "yue";
            return parts;
        }
        Parts result;
        for (auto &part : parts) {
            part.language = part.language == "en" ? "en" : mode;
            append(result, std::move(part));
        }
        return result;
    }
};
LanguageRouter::LanguageRouter(const std::string &directory) : impl_(std::make_unique<Impl>(directory)) {}
LanguageRouter::~LanguageRouter() = default;
std::vector<LanguageRouter::Span> LanguageRouter::split(std::u32string_view input, const std::string &mode) const {
    return impl_->split(input, mode);
}
} // namespace tts::text
