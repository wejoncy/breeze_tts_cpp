#pragma once
#include "io/tokenizer.h"
#include "execution/request_planner.h"
#include "text/pinyin.h"
#include <memory>
#include <utility>

namespace tts::text {
struct G2PWTask final : PreparationTask {
    std::vector<int> ids, positions, characters, labels;
    std::vector<int> offsets{0};
};
// Immutable pronunciation assets and CPU tokenization. Network execution is
// transferred to the same GPU executor as every other conditioning stage.
class G2PW {
    struct Impl;
    std::unique_ptr<Impl> impl_;
    const PypinyinResolver &fallback_;
  public:
    class Resolver final : public PinyinResolver {
        const G2PW &owner_;
        struct Batch {
            std::unique_ptr<G2PWTask> task = std::make_unique<G2PWTask>();
            std::vector<std::pair<size_t, size_t>> targets;
        };
        std::vector<std::pair<std::u32string, std::vector<std::string>>> readings_;
        std::vector<Batch> batches_;
        size_t current_ = 0;
        bool pending_ = false;
      public:
        Resolver(const G2PW &, const std::vector<std::u32string> &segments);
        std::unique_ptr<PreparationTask> next();
        void accept(std::unique_ptr<PreparationTask>);
        void resolve(const char32_t *word, size_t count,
                     std::vector<PinyinSyl> *out) const override {
            owner_.fallback_.resolve(word, count, out);
        }
        bool resolveSentenceTone3(const char32_t *word, size_t count,
                                  std::vector<std::string> *out) const override;
        bool isNumeric(uint32_t cp) const override { return owner_.fallback_.isNumeric(cp); }
    };
    G2PW(const std::string &directory, const Tokenizer &,
          const PypinyinResolver &);
    ~G2PW();
    Resolver resolver(const std::vector<std::u32string> &segments) const { return {*this, segments}; }
};
} // namespace tts::text
