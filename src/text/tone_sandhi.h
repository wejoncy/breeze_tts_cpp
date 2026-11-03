#pragma once
#include "text/weighted_segmenter.h"
#include <memory>
#include <string_view>
namespace tts::text {
class PypinyinResolver;
class ToneSandhi {
    struct Impl;
    std::unique_ptr<Impl> impl_;
  public:
    ToneSandhi(const std::string &directory, const WeightedSegmenter &, const PypinyinResolver &);
    ~ToneSandhi();
    std::vector<SegToken> premerge(std::vector<SegToken> segments) const;
    void modify(std::u32string_view word, std::string_view pos, std::vector<std::string> &finals) const;
};
} // namespace tts::text
