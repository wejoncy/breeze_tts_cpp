#pragma once
#include <stdexcept>
#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <memory>
extern "C" {
struct HFToken {
    uint32_t id;
    size_t start, end;
};
void *hf_tokenizer_new(const char *, char *, size_t);
void hf_tokenizer_free(void *);
int hf_tokenizer_encode(void *, const char *, bool, HFToken **, size_t *, char *, size_t);
void hf_tokens_free(HFToken *, size_t);
}
namespace tts {
class Tokenizer {
    void *impl_ = nullptr;

  public:
    explicit Tokenizer(const std::string &path) {
        char err[1024]{};
        impl_ = hf_tokenizer_new(path.c_str(), err, sizeof(err));
        if (!impl_)
            throw std::runtime_error(err);
    }
    ~Tokenizer() {
        hf_tokenizer_free(impl_);
    }
    Tokenizer(const Tokenizer &) = delete;
    Tokenizer &operator=(const Tokenizer &) = delete;
    std::vector<HFToken> tokenize(const std::string &text, bool bos = true) const {
        HFToken *tokens = nullptr;
        size_t n = 0;
        char err[1024]{};
        if (hf_tokenizer_encode(impl_, text.c_str(), bos, &tokens, &n, err, sizeof(err)))
            throw std::runtime_error(err);
        auto release = [n](HFToken *p) { hf_tokens_free(p, n); };
        std::unique_ptr<HFToken, decltype(release)> owned(tokens, release);
        return n ? std::vector<HFToken>(tokens, tokens + n) : std::vector<HFToken>{};
    }
    std::vector<int> encode(const std::string &text, bool bos = true) const {
        auto tokens = tokenize(text, bos);
        std::vector<int> out;
        out.reserve(tokens.size());
        for (auto token : tokens) out.push_back(int(token.id));
        return out;
    }
};
} // namespace tts
