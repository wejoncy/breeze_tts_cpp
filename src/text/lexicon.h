#pragma once
#include "io/json.h"
#include "text/utf8.h"
#include <map>

namespace tts::text {
template <typename Value>
using Lexicon = std::map<std::u32string, std::vector<Value>, std::less<>>;

template <typename Value> Lexicon<Value> load_lexicon(const Json &data) {
    if (!data.is_object()) throw std::invalid_argument("pronunciation dictionary must be an object");
    Lexicon<Value> result;
    for (const auto &[word, values] : data.items()) {
        auto key = decode(word);
        if (key.empty()) throw std::invalid_argument("empty pronunciation dictionary key");
        result.emplace(std::move(key), values.get<std::vector<Value>>());
    }
    return result;
}
// Match complete keys without imposing a model-specific maximum word length.
template <typename Value>
auto longest_match(const Lexicon<Value> &dictionary, std::u32string_view text) {
    auto selected = dictionary.end();
    for (size_t length = 1; length <= text.size(); ++length) {
        auto prefix = text.substr(0, length);
        auto candidate = dictionary.lower_bound(prefix);
        if (candidate == dictionary.end() ||
            std::u32string_view(candidate->first).substr(0, length) != prefix) break;
        if (candidate->first.size() == length) selected = candidate;
    }
    return selected;
}
} // namespace tts::text
