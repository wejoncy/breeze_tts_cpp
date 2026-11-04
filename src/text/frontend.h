#pragma once
#include "text/g2pw.h"
#include "text/phonemizer.h"
#include "text/english.h"
#include "text/cantonese.h"
#include "text/korean.h"
#include "text/japanese.h"
#include "text/language_router.h"
#include <optional>

namespace tts::text {
struct TextInput {
    std::string normalized;
    std::vector<int> phones, bert_ids, phone_bert_rows;
    // Independent Chinese BERT sequences. Other languages use row -1 and do
    // not enter BERT. Offsets address the concatenated IDs/projection rows.
    std::vector<int> bert_offsets{0};
};
// CPU linguistic work lives on the scheduler. Only IDs/alignment cross into
// the executor, where the model's BERT network produces GPU conditioning.
class TextFrontend {
    Tokenizer bert_tokenizer_, pronunciation_tokenizer_;
    text::ChinesePhonemizer chinese_;
    text::EnglishPhonemizer english_;
    std::unique_ptr<text::CantonesePhonemizer> cantonese_;
    std::unique_ptr<text::KoreanPhonemizer> korean_;
    std::unique_ptr<text::JapanesePhonemizer> japanese_;
    std::unique_ptr<text::G2PW> pronunciation_;
    std::unique_ptr<LanguageRouter> router_;
    int max_bert_tokens_, english_pause_;

  public:
    enum class Language { Chinese, English, Cantonese, Korean, Japanese };
    struct Span { std::u32string normalized; Language language; };
  private:
    std::vector<Span> normalize(const std::string &, const std::string &) const;
  public:
    class Preparation {
        const TextFrontend &owner_;
        std::string input_, language_;
        std::vector<Span> spans_;
        std::unique_ptr<text::G2PW::Resolver> resolver_;
        std::optional<TextInput> result_;
        bool retried_ = false;
        void begin_pass();
        TextInput collect();
      public:
        Preparation(const TextFrontend &, std::string input, std::string language);
        std::unique_ptr<PreparationTask> next();
        void accept(std::unique_ptr<PreparationTask> task) { resolver_->accept(std::move(task)); }
        TextInput finish();
    };
    TextFrontend(const std::string &directory, int max_bert_tokens);
    size_t max_characters() const { return size_t(max_bert_tokens_ - 2); }
    std::unique_ptr<Preparation> begin(const std::string &, const std::string &language = {}) const;
    TextInput prepare(const std::string &, const PreparationExecutor &execute,
                       const std::string &language = {}) const;
};
} // namespace tts::text
