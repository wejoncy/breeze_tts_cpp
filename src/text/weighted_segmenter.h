#pragma once
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace tts::text {
struct SegToken { std::u32string word; std::string flag; };
// Exact checkpoint dictionary route and POS/finalseg HMM execution. Neither
// WordPiece boundaries nor per-character greedy matching replace word analysis.
class WeightedSegmenter {
    struct Impl;
    std::unique_ptr<Impl> impl_;
  public:
    explicit WeightedSegmenter(const std::string &directory);
    ~WeightedSegmenter();
    std::vector<SegToken> segment(std::u32string_view) const;
    std::vector<std::u32string> search(std::u32string_view) const;
};
} // namespace tts::text
