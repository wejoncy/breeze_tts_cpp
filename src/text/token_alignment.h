#pragma once
#include "io/tokenizer.h"

namespace tts::text {
// Return the first token covering a UTF-8 byte offset, retaining original
// indices and ignoring empty special-token spans. Ordered offsets take one
// forward pass; other tokenizer layouts retain first-match search semantics.
class TokenSpanCursor {
    const std::vector<HFToken> &tokens_;
    size_t cursor_ = 0, previous_ = 0;
    bool ordered_ = true;
  public:
    explicit TokenSpanCursor(const std::vector<HFToken> &tokens) : tokens_(tokens) {
        size_t start = 0;
        for (const auto &token : tokens) if (token.start < token.end) {
            if (token.start < start) ordered_ = false;
            start = token.start;
        }
    }
    size_t find(size_t byte) {
        if (!ordered_ || byte < previous_) cursor_ = 0;
        previous_ = byte;
        while (cursor_ < tokens_.size()) {
            const auto &token = tokens_[cursor_];
            if (token.start <= byte && byte < token.end) return cursor_;
            if (ordered_ && token.start < token.end && token.start > byte) break;
            ++cursor_;
        }
        return tokens_.size();
    }
};
} // namespace tts::text
