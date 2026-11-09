#pragma once
#include "io/tokenizer.h"
#include "execution/input_plan.h"
#include "text/pinyin.h"
#include <memory>
#include <utility>

namespace tts::text {
struct G2PWTask final : PreparationTask {
    std::vector<int> ids, positions, characters;
    Tensor labels;
    std::vector<int> offsets{0};
    std::string_view program() const override { return "pronunciation"; }
    InputBatch input() const override;
    std::vector<PreparationOutput> outputs() const override { return {{"labels", int(positions.size())}}; }
    void accept(HostResults) override;
};
// Immutable pronunciation assets and CPU tokenization. Network execution is
// transferred by Engine to the runner like every other conditioning stage.
class G2PW {
    struct Impl;
    std::unique_ptr<Impl> impl_;
    const PypinyinResolver &fallback_;
  public:
    class Resolver {
        const G2PW &owner_;
        struct Batch {
            std::unique_ptr<G2PWTask> task = std::make_unique<G2PWTask>();
            std::vector<std::pair<size_t, size_t>> targets;
        };
        SentenceReadings readings_;
        std::vector<Batch> batches_;
        size_t current_ = 0;
        bool pending_ = false;
      public:
        Resolver(const G2PW &, const std::vector<std::u32string> &segments);
        std::unique_ptr<PreparationTask> next();
        void accept(std::unique_ptr<PreparationTask>);
        const SentenceReadings &readings() const;
    };
    G2PW(const std::string &directory, const Tokenizer &,
          const PypinyinResolver &);
    ~G2PW();
    Resolver resolver(const std::vector<std::u32string> &segments) const { return {*this, segments}; }
};
} // namespace tts::text
