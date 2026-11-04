// TTS routing normalization around the upstream FastText classifier.
#include "text/language_detector.h"
#include "text/utf8.h"
#include "io/json.h"
#include <fasttext.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>

namespace tts::text {
namespace {
using Ranges = std::vector<std::pair<char32_t, char32_t>>;
Ranges ranges(const Json &data) {
    Ranges result;
    for (const auto &entry : data) {
        char32_t first = entry.at(0), last = entry.at(1);
        if (last < first || (!result.empty() && first <= result.back().second))
            throw std::invalid_argument("invalid language-detector Unicode ranges");
        result.emplace_back(first, last);
    }
    return result;
}
bool contains(const Ranges &ranges, char32_t cp) {
    auto found = std::upper_bound(ranges.begin(), ranges.end(), cp,
        [](char32_t value, auto range) { return value < range.first; });
    return found != ranges.begin() && cp <= (--found)->second;
}
} // namespace
struct LanguageDetector::Impl final : private fasttext::FastText {
    int max_input;
    bool normalize_input;
    Ranges upper, cased, ignorable;
    std::map<char32_t, std::u32string> lower;
    explicit Impl(const std::string &directory) {
        const auto data = read_json(directory + "/language_policy.json");
        max_input = data.at("max_input_length");
        normalize_input = data.at("normalize_input");
        if (max_input < 1) throw std::invalid_argument("invalid language-detector input limit");
        upper = ranges(data.at("upper")); cased = ranges(data.at("cased"));
        ignorable = ranges(data.at("case_ignorable"));
        for (const auto &[from, to] : data.at("lower").items()) {
            auto codepoint = decode(from);
            if (codepoint.size() != 1) throw std::invalid_argument("invalid Unicode lowercase key");
            lower.emplace(codepoint.front(), decode(to.get<std::string>()));
        }
        const auto path = std::filesystem::u8path(directory) / "language_detector.bin";
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("cannot open language classifier: " + path.u8string());
        if (!checkModel(input)) throw std::invalid_argument("invalid FastText model: " + path.u8string());
        loadModel(input);
        if (getArgs().model != fasttext::model_name::sup)
            throw std::invalid_argument("language detection requires a supervised FastText model");
    }
    std::u32string normalize(std::u32string_view text) const {
        std::u32string value(text.substr(0, max_input));
        std::replace(value.begin(), value.end(), U'\n', U' ');
        if (!normalize_input) return value;
        size_t uppercase = 0, letters = 0;
        bool all_upper = true, any_cased = false;
        for (auto cp : value) {
            if (contains(cased, cp)) { any_cased = true; if (!contains(upper, cp)) all_upper = false; }
            if (cp >= U'A' && cp <= U'Z') ++uppercase;
            if ((cp >= U'A' && cp <= U'Z') || (cp >= U'a' && cp <= U'z')) ++letters;
        }
        if (!(all_upper && any_cased) && !(value.size() > 5 && uppercase > 0.8 * letters)) return value;
        std::u32string result;
        for (size_t i = 0; i < value.size(); ++i) {
            auto cp = value[i];
            if (cp == U'Σ') {
                size_t before = i, after = i + 1;
                while (before && contains(ignorable, value[before - 1])) --before;
                while (after < value.size() && contains(ignorable, value[after])) ++after;
                const bool final = before && contains(cased, value[before - 1]) &&
                    (after == value.size() || !contains(cased, value[after]));
                result += final ? U'ς' : U'σ';
            } else if (auto found = lower.find(cp); found != lower.end()) result += found->second;
            else result += cp;
        }
        return result;
    }
    std::vector<LanguageDetector::Candidate> detect(std::u32string_view text, int count, float threshold) const {
        if (count < 1 || !std::isfinite(threshold) || threshold < 0 || threshold > 1)
            throw std::invalid_argument("invalid language-detector prediction parameters");
        std::istringstream input(encode(normalize(text)) + "\n");
        std::vector<std::pair<fasttext::real, std::string>> predictions;
        predictLine(input, predictions, count, threshold);
        std::vector<LanguageDetector::Candidate> result;
        result.reserve(predictions.size());
        for (auto &prediction : predictions) {
            if (prediction.second.compare(0, 9, "__label__"))
                throw std::invalid_argument("language classifier has an unsupported label prefix");
            result.push_back({prediction.second.substr(9), std::min(prediction.first, 1.f)});
        }
        return result;
    }
};
LanguageDetector::LanguageDetector(const std::string &directory) : impl_(std::make_unique<Impl>(directory)) {}
LanguageDetector::~LanguageDetector() = default;
std::vector<LanguageDetector::Candidate> LanguageDetector::detect(std::u32string_view text, int count, float threshold) const {
    return impl_->detect(text, count, threshold);
}
} // namespace tts::text
