// Native FastText feature extraction and hierarchical softmax (MIT, Facebook).
#include "text/language_detector.h"
#include "text/utf8.h"
#include "io/json.h"
#include "io/safetensors.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <unordered_map>

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
uint32_t hash(std::string_view text) {
    uint32_t h = 2166136261;
    for (auto cp : text) { h ^= uint32_t(int8_t(cp)); h *= 16777619; }
    return h;
}
bool whitespace(char cp) {
    return cp == ' ' || cp == '\n' || cp == '\r' || cp == '\t' ||
           cp == '\v' || cp == '\f' || cp == '\0';
}
float log_probability(float value) { return std::log(value + 1e-5); }
} // namespace
struct LanguageDetector::Impl {
    SafeTensors weights;
    const float *input, *output;
    int dimensions, bucket, minimum, maximum, word_ngrams, max_input;
    bool normalize_input;
    std::unordered_map<std::string, int> words;
    std::vector<std::string> labels;
    std::vector<std::pair<int, int>> tree;
    Ranges upper, cased, ignorable;
    std::map<char32_t, std::u32string> lower;
    explicit Impl(const std::string &directory) {
        auto data = read_json(directory + "/language_detector.json");
        weights.open_file(directory + "/language_detector.safetensors");
        dimensions = data.at("dimensions"); bucket = data.at("bucket");
        minimum = data.at("minn"); maximum = data.at("maxn");
        word_ngrams = data.at("word_ngrams"); max_input = data.at("max_input_length");
        normalize_input = data.at("normalize_input");
        if (dimensions < 1 || bucket < 1 || minimum < 1 || maximum < minimum ||
            word_ngrams < 1 || max_input < 1)
            throw std::invalid_argument("invalid language-detector dimensions");
        const auto vocabulary = data.at("words").get<std::vector<std::string>>();
        words.reserve(vocabulary.size());
        for (const auto &word : vocabulary)
            if (!words.emplace(word, int(words.size())).second)
                throw std::invalid_argument("duplicate language-detector dictionary word");
        labels = data.at("labels").get<std::vector<std::string>>();
        const auto &wi = weights.get("detector.input"), &wo = weights.get("detector.output");
        if (words.empty() || labels.empty() || words.size() > size_t(INT32_MAX - bucket) ||
            wi.type != Type::F32 || wo.type != Type::F32 ||
            wi.shape != std::vector<int>{int(words.size()) + bucket, dimensions} ||
            wo.shape != std::vector<int>{int(labels.size()), dimensions})
            throw std::invalid_argument("language-detector matrix/dictionary mismatch");
        input = static_cast<const float *>(wi.data); output = static_cast<const float *>(wo.data);
        for (const auto &entry : data.at("tree")) tree.emplace_back(entry.at(0), entry.at(1));
        if (tree.size() != 2 * labels.size() - 1)
            throw std::invalid_argument("invalid language-detector tree size");
        for (size_t i = 0; i < tree.size(); ++i) {
            auto [left, right] = tree[i];
            if (i < labels.size() ? left != -1 || right != -1 :
                left < 0 || right < 0 || size_t(left) >= i || size_t(right) >= i)
                throw std::invalid_argument("invalid language-detector tree children");
        }
        upper = ranges(data.at("upper")); cased = ranges(data.at("cased"));
        ignorable = ranges(data.at("case_ignorable"));
        for (const auto &[from, to] : data.at("lower").items()) {
            auto cp = decode(from);
            if (cp.size() != 1) throw std::invalid_argument("invalid Unicode lowercase key");
            lower.emplace(cp.front(), decode(to.get<std::string>()));
        }
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
    void subwords(const std::string &token, std::vector<int> &features) const {
        const std::string text = "<" + token + ">";
        for (size_t first = 0; first < text.size(); ++first) {
            if ((uint8_t(text[first]) & 0xc0) == 0x80) continue;
            size_t end = first;
            for (int length = 1; end < text.size() && length <= maximum; ++length) {
                ++end;
                while (end < text.size() && (uint8_t(text[end]) & 0xc0) == 0x80) ++end;
                if (length >= minimum && !(length == 1 && (!first || end == text.size())))
                    features.push_back(int(words.size()) + int(hash(std::string_view(text).substr(first, end - first)) % bucket));
            }
        }
    }
    std::vector<int> features(const std::string &text) const {
        std::vector<int> features;
        std::vector<int32_t> hashes;
        auto add = [&](const std::string &token) {
            auto found = words.find(token);
            if (found == words.end() && token.compare(0, 9, "__label__") == 0) return;
            if (found != words.end()) features.push_back(found->second);
            if (token != "</s>") subwords(token, features);
            hashes.push_back(int32_t(hash(token)));
        };
        size_t cursor = 0;
        bool end_sentence = false;
        while (cursor < text.size()) {
            while (cursor < text.size() && whitespace(text[cursor])) ++cursor;
            if (cursor == text.size()) break;
            size_t end = cursor;
            while (end < text.size() && !whitespace(text[end])) ++end;
            auto token = text.substr(cursor, end - cursor);
            add(token);
            if (token == "</s>") { end_sentence = true; break; }
            cursor = end;
        }
        if (!end_sentence) add("</s>");
        for (size_t i = 0; i < hashes.size(); ++i) {
            uint64_t h = uint64_t(int64_t(hashes[i]));
            for (size_t j = i + 1; j < hashes.size() && j - i < size_t(word_ngrams); ++j) {
                h = h * 116049371 + uint64_t(int64_t(hashes[j]));
                features.push_back(int(words.size()) + int(h % bucket));
            }
        }
        return features;
    }
    std::vector<LanguageDetector::Candidate> detect(std::u32string_view text, int k, float threshold) const {
        if (k < 1 || !std::isfinite(threshold) || threshold < 0 || threshold > 1)
            throw std::invalid_argument("invalid language-detector prediction parameters");
        auto ids = features(encode(normalize(text)));
        if (ids.empty()) return {};
        std::vector<float> hidden(dimensions, 0);
        for (auto id : ids)
            for (int j = 0; j < dimensions; ++j) hidden[j] += input[size_t(id) * dimensions + j];
        const float factor = 1.0 / ids.size();
        for (auto &value : hidden) value *= factor;
        using Prediction = std::pair<float, int>;
        std::vector<Prediction> heap, stack{{0, int(tree.size()) - 1}};
        auto compare = [](auto a, auto b) { return a.first > b.first; };
        const float minimum_score = log_probability(threshold);
        while (!stack.empty()) {
            auto [score, node] = stack.back(); stack.pop_back();
            if (score < minimum_score || (heap.size() == size_t(k) && score < heap.front().first)) continue;
            const auto [left, right] = tree[node];
            if (left < 0) {
                heap.emplace_back(score, node);
                std::push_heap(heap.begin(), heap.end(), compare);
                if (heap.size() > size_t(k)) { std::pop_heap(heap.begin(), heap.end(), compare); heap.pop_back(); }
                continue;
            }
            float dot = 0;
            for (int j = 0; j < dimensions; ++j) dot += output[size_t(node - labels.size()) * dimensions + j] * hidden[j];
            float probability = 1. / (1 + std::exp(-dot));
            stack.emplace_back(score + log_probability(probability), right);
            stack.emplace_back(score + log_probability(1.0 - probability), left);
        }
        std::sort_heap(heap.begin(), heap.end(), compare);
        std::vector<LanguageDetector::Candidate> result;
        for (auto [score, label] : heap) result.push_back({labels[label], std::min(std::exp(score), 1.f)});
        return result;
    }
};
LanguageDetector::LanguageDetector(const std::string &directory) : impl_(std::make_unique<Impl>(directory)) {}
LanguageDetector::~LanguageDetector() = default;
std::vector<LanguageDetector::Candidate> LanguageDetector::detect(std::u32string_view text, int count, float threshold) const {
    return impl_->detect(text, count, threshold);
}
} // namespace tts::text
