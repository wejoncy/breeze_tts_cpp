#include "text/jieba.h"
#include "text/utf8.h"
#include <cppjieba/MixSegment.hpp>
#include <cppjieba/QuerySegment.hpp>

namespace tts::text {
struct JiebaSegmenter::Impl {
    cppjieba::DictTrie dictionary;
    cppjieba::HMMModel model;
    cppjieba::MixSegment tagged;
    cppjieba::QuerySegment query;
    explicit Impl(const std::string &directory)
        : dictionary(directory + "/jieba.dict.utf8"), model(directory + "/jieba.hmm.utf8"),
          tagged(&dictionary, &model), query(&dictionary, &model) {}
};
JiebaSegmenter::JiebaSegmenter(const std::string &directory) : impl_(std::make_unique<Impl>(directory)) {}
JiebaSegmenter::~JiebaSegmenter() = default;
std::vector<SegToken> JiebaSegmenter::segment(std::u32string_view text) const {
    std::vector<std::pair<std::string, std::string>> words;
    impl_->tagged.Tag(encode(text), words);
    std::vector<SegToken> result;
    result.reserve(words.size());
    for (auto &word : words) result.push_back({decode(word.first), std::move(word.second)});
    return result;
}
std::vector<std::u32string> JiebaSegmenter::search(std::u32string_view text) const {
    std::vector<std::string> words;
    impl_->query.Cut(encode(text), words);
    std::vector<std::u32string> result;
    result.reserve(words.size());
    for (const auto &word : words) result.push_back(decode(word));
    return result;
}
} // namespace tts::text
