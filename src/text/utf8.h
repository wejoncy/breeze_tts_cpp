#pragma once
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace tts::text {
inline std::u32string decode(std::string_view input, std::vector<size_t> *offsets = nullptr) {
    std::u32string result;
    if (offsets) offsets->clear();
    for (size_t i = 0; i < input.size();) {
        if (offsets) offsets->push_back(i);
        unsigned lead = static_cast<unsigned char>(input[i++]);
        unsigned count = lead < 0x80 ? 0 : lead >= 0xc2 && lead < 0xe0 ? 1 :
                         lead < 0xf0 && lead >= 0xe0 ? 2 : lead >= 0xf0 && lead < 0xf5 ? 3 : 4;
        if (count > 3 || input.size() - i < count) throw std::invalid_argument("invalid UTF-8 text");
        char32_t cp = count ? lead & ((1u << (6 - count)) - 1) : lead;
        for (unsigned j = 0; j < count; ++j) {
            unsigned next = static_cast<unsigned char>(input[i++]);
            if ((next & 0xc0) != 0x80) throw std::invalid_argument("invalid UTF-8 continuation");
            cp = (cp << 6) | (next & 0x3f);
        }
        if ((count && cp < (count == 1 ? 0x80u : count == 2 ? 0x800u : 0x10000u)) ||
            (cp >= 0xd800 && cp <= 0xdfff) || cp > 0x10ffff)
            throw std::invalid_argument("invalid UTF-8 codepoint");
        result.push_back(cp);
    }
    if (offsets) offsets->push_back(input.size());
    return result;
}
inline std::string encode(std::u32string_view input) {
    std::string result;
    for (char32_t cp : input) {
        if (cp < 0x80) result.push_back(char(cp));
        else if (cp < 0x800) {
            result.push_back(char(0xc0 | (cp >> 6)));
            result.push_back(char(0x80 | (cp & 0x3f)));
        } else if (cp < 0x10000 && !(cp >= 0xd800 && cp <= 0xdfff)) {
            result.push_back(char(0xe0 | (cp >> 12)));
            result.push_back(char(0x80 | ((cp >> 6) & 0x3f)));
            result.push_back(char(0x80 | (cp & 0x3f)));
        } else if (cp >= 0x10000 && cp <= 0x10ffff) {
            result.push_back(char(0xf0 | (cp >> 18)));
            result.push_back(char(0x80 | ((cp >> 12) & 0x3f)));
            result.push_back(char(0x80 | ((cp >> 6) & 0x3f)));
            result.push_back(char(0x80 | (cp & 0x3f)));
        } else throw std::invalid_argument("invalid Unicode codepoint");
    }
    return result;
}
} // namespace tts::text
