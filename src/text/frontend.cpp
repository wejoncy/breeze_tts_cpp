#include "text/frontend.h"
#include "text/frontend_assets.h"
#include "text/utf8.h"
#include "text/token_alignment.h"
#include <algorithm>
#include <numeric>

namespace tts::text {
TextFrontend::TextFrontend(const std::string &directory, int max_tokens)
    : bert_tokenizer_(directory + "/tokenizer.json"),
      pronunciation_tokenizer_(directory + "/g2pw_tokenizer.json"), chinese_(directory),
      english_(directory), max_bert_tokens_(max_tokens),
      english_pause_(english_.pronounce(",").at(0)) {
    if (max_tokens < 3) throw std::invalid_argument("BERT context must include special tokens");
    pronunciation_ = std::make_unique<text::G2PW>(directory, pronunciation_tokenizer_, chinese_.dictionary());
    const FrontendAssets assets{ModelLoader(directory)};
    if (assets.available("yue"))
        cantonese_ = std::make_unique<text::CantonesePhonemizer>(directory, chinese_.dictionary());
    if (assets.available("ko")) korean_ = std::make_unique<text::KoreanPhonemizer>(directory);
    if (assets.available("ja")) japanese_ = std::make_unique<text::JapanesePhonemizer>(directory);
    if (assets.available("auto")) router_ = std::make_unique<LanguageRouter>(directory);
}
namespace {
bool latin(char32_t c) {
    return (c >= U'a' && c <= U'z') || (c >= U'A' && c <= U'Z') ||
           (c >= 0xc0 && c <= 0x24f);
}
bool ideograph(char32_t c) {
    return (c >= 0x3400 && c <= 0x9fff) || (c >= 0x20000 && c <= 0x323af);
}
std::vector<std::u32string> pronunciation_segments(const text::ChinesePhonemizer &chinese,
                                                  const std::vector<TextFrontend::Span> &spans) {
    std::vector<std::u32string> result;
    for (const auto &span : spans) if (span.language == TextFrontend::Language::Chinese) {
        auto segments = chinese.segments(span.normalized);
        result.insert(result.end(), segments.begin(), segments.end());
    }
    return result;
}
} // namespace
std::vector<TextFrontend::Span> TextFrontend::normalize(const std::string &input,
                                               const std::string &language) const {
    const bool automatic = language == "auto" || language == "auto_yue";
    const bool japanese = language == "ja" || language == "all_ja";
    const bool cantonese = language == "yue" || language == "all_yue";
    const bool korean = language == "ko" || language == "all_ko";
    if (!language.empty() && language != "zh" && language != "all_zh" && language != "en" &&
        !cantonese && !korean && !japanese && !automatic)
        throw std::invalid_argument("unsupported GPT-SoVITS text language: " + language);
    auto require = [&](Language selected) {
        if (selected == Language::Cantonese && !cantonese_)
            throw std::invalid_argument("Cantonese assets are absent from this checkpoint; reconvert with --cantonese-jyutping");
        if (selected == Language::Korean && !korean_)
            throw std::invalid_argument("Korean assets are absent from this checkpoint; reconvert with --korean-g2p, --korean-dictionary and --korean-jamo");
        if (selected == Language::Japanese && !japanese_)
            throw std::invalid_argument("Japanese assets are absent from this checkpoint; reconvert with --japanese-dictionary and --japanese-user-dictionary");
    };
    if (cantonese) require(Language::Cantonese);
    if (korean) require(Language::Korean);
    if (japanese) require(Language::Japanese);
    if (automatic && !router_)
        throw std::invalid_argument("Automatic language routing assets are absent from this checkpoint; reconvert with --language-detector, --language-splitter and --language-budoux");
    auto codepoints = text::decode(input);
    codepoints.erase(std::unique(codepoints.begin(), codepoints.end(), [](char32_t a, char32_t b) {
        return a == U' ' && b == U' ';
    }), codepoints.end());
    std::vector<Span> spans;
    const auto primary = cantonese ? Language::Cantonese : korean ? Language::Korean : japanese ? Language::Japanese : Language::Chinese;
    auto append = [&](std::u32string value, Language selected) {
        require(selected);
        std::u32string normalized;
        switch (selected) {
        case Language::English: normalized = text::decode(english_.normalize(text::encode(value))); break;
        case Language::Chinese: normalized = chinese_.normalize(value); break;
        case Language::Cantonese: normalized = cantonese_->normalize(value); break;
        case Language::Korean: normalized = std::move(value); break;
        case Language::Japanese: normalized = japanese_->normalize(value); break;
        }
        if (!normalized.empty() || selected == Language::English)
            spans.push_back({std::move(normalized), selected});
    };
    if (language == "en") append(std::move(codepoints), Language::English);
    else if (router_) {
        for (auto part : router_->split(codepoints, language.empty() ? "zh" : language)) {
            if (part.language == "zh") append(std::move(part.text), Language::Chinese);
            else if (part.language == "en") append(std::move(part.text), Language::English);
            else if (part.language == "yue") {
                append(std::move(part.text), Language::Cantonese);
            } else if (part.language == "ko") {
                append(std::move(part.text), Language::Korean);
            } else if (part.language == "ja") {
                append(std::move(part.text), Language::Japanese);
            } else throw std::invalid_argument("Unsupported detected language: " + part.language);
        }
    } else {
        // Explicit Chinese/Cantonese modes keep Latin runs in the English frontend.
        // Neutral punctuation/numbers stay with their preceding language; a
        // leading neutral run follows the first spoken script.
        size_t start = 0;
        int current = -1;
        for (size_t i = 0; i < codepoints.size(); ++i) {
            char32_t c = codepoints[i];
            const bool hangul = (c >= 0x1100 && c <= 0x11ff) ||
                                (c >= 0x3130 && c <= 0x318f) ||
                                (c >= 0xac00 && c <= 0xd7ff) ||
                                (c >= 0xa960 && c <= 0xa97f);
            int script = latin(c) ? 1 : ideograph(c) || (korean && hangul) ? 0 : -1;
            const bool kana = c >= 0x3040 && c <= 0x30ff;
            if (japanese && kana) script = 0;
            if ((kana && !japanese) || (hangul && !korean))
                throw std::invalid_argument("text contains a script outside the selected language frontend");
            if (script < 0) continue;
            if (current >= 0 && current != script) {
                append(codepoints.substr(start, i - start), current == 1 ? Language::English : primary);
                start = i;
            }
            current = script;
        }
        append(codepoints.substr(start), current == 1 ? Language::English : primary);
    }
    return spans;
}
std::unique_ptr<TextFrontend::Preparation> TextFrontend::begin(const std::string &input,
                                                      const std::string &language) const {
    return std::make_unique<Preparation>(*this, input, language);
}
TextFrontend::Preparation::Preparation(const TextFrontend &owner, std::string input, std::string language)
    : owner_(owner), input_(std::move(input)), language_(std::move(language)) { begin_pass(); }
void TextFrontend::Preparation::begin_pass() {
    spans_ = owner_.normalize(input_, language_);
    resolver_ = std::make_unique<text::G2PW::Resolver>(
        owner_.pronunciation_->resolver(pronunciation_segments(owner_.chinese_, spans_)));
}
std::unique_ptr<PreparationTask> TextFrontend::Preparation::next() {
    while (!result_) {
        if (auto task = resolver_->next()) return task;
        auto result = collect();
        // Match upstream's one-time short-input retry through the full frontend,
        // including new BERT/G2PW positions and any asynchronous model tasks.
        if (!retried_ && result.phones.size() < 6) {
            input_.insert(0, ".");
            retried_ = true;
            begin_pass();
            continue;
        }
        if (result.phones.empty()) throw std::invalid_argument("text has no speakable phonemes");
        result_.emplace(std::move(result));
    }
    return {};
}
TextInput TextFrontend::Preparation::finish() {
    if (!result_) throw std::logic_error("text preparation has not completed");
    return std::move(*result_);
}
TextInput TextFrontend::Preparation::collect() {
    TextInput result;
    for (const auto &span : spans_) {
        const auto normalized_text = text::encode(span.normalized);
        result.normalized += normalized_text;
        if (span.language != Language::Chinese) {
            std::vector<int> phones;
            switch (span.language) {
            case Language::English: phones = owner_.english_.pronounce(normalized_text); break;
            case Language::Cantonese: phones = owner_.cantonese_->pronounce(span.normalized); break;
            case Language::Korean: phones = owner_.korean_->pronounce(span.normalized); break;
            case Language::Japanese: phones = owner_.japanese_->pronounce(span.normalized); break;
            case Language::Chinese: throw std::logic_error("unexpected Chinese frontend branch");
            }
            if (span.language == Language::English && phones.size() < 4)
                phones.insert(phones.begin(), owner_.english_pause_);
            result.phones.insert(result.phones.end(), phones.begin(), phones.end());
            result.phone_bert_rows.insert(result.phone_bert_rows.end(), phones.size(), -1);
            continue;
        }
        auto phones = owner_.chinese_.pronounce(span.normalized, resolver_->readings());
        std::vector<size_t> offsets;
        auto normalized = text::decode(normalized_text, &offsets);
        if (phones.word2ph.size() != normalized.size() ||
            std::accumulate(phones.word2ph.begin(), phones.word2ph.end(), size_t(0)) !=
                phones.phones.size())
            throw std::invalid_argument("phoneme and normalized-text alignment mismatch");
        auto tokens = owner_.bert_tokenizer_.tokenize(normalized_text);
        if (tokens.size() > size_t(owner_.max_bert_tokens_))
            throw std::invalid_argument("text segment exceeds BERT context");
        int base = int(result.bert_ids.size());
        for (auto token : tokens) result.bert_ids.push_back(int(token.id));
        result.bert_offsets.push_back(int(result.bert_ids.size()));
        result.phones.insert(result.phones.end(), phones.phones.begin(), phones.phones.end());
        text::TokenSpanCursor alignment(tokens);
        for (size_t i = 0; i < normalized.size(); ++i) {
            auto found = alignment.find(offsets[i]);
            if (found == tokens.size())
                throw std::invalid_argument("Chinese character has no BERT span");
            result.phone_bert_rows.insert(result.phone_bert_rows.end(), phones.word2ph[i],
                                          base + int(found));
        }
    }
    return result;
}
} // namespace tts::text
