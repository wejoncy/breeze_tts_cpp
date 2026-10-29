#pragma once
#include <stdexcept>
#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
extern "C" {
void *hf_tokenizer_new(const char *, char *, size_t);
void hf_tokenizer_free(void *);
int hf_tokenizer_encode(void *, const char *, bool, uint32_t **, size_t *, char *, size_t);
void hf_tokens_free(uint32_t *, size_t);
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
    std::vector<int> encode(const std::string &text, bool bos = true) const {
        uint32_t *ids = nullptr;
        size_t n = 0;
        char err[1024]{};
        if (hf_tokenizer_encode(impl_, text.c_str(), bos, &ids, &n, err, sizeof(err)))
            throw std::runtime_error(err);
        std::vector<int> out(ids, ids + n);
        hf_tokens_free(ids, n);
        return out;
    }
};
} // namespace tts
