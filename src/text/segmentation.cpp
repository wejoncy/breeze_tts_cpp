#include "text/segmentation.h"
#include "text/utf8.h"
#include <algorithm>

namespace tts::text {
namespace {
bool contains(std::u32string_view set, char32_t c) {
    return set.find(c) != std::u32string_view::npos;
}
constexpr std::u32string_view boundaries = U"，。？！,.?!~:：—…";
constexpr std::u32string_view punctuation = U"!?…,. -";
bool digit(char32_t c) { return c >= U'0' && c <= U'9'; }
bool spoken(std::u32string_view s) {
    return std::any_of(s.begin(), s.end(), [](char32_t c) {
        return (c >= U'0' && c <= U'9') || (c >= U'A' && c <= U'Z') ||
               (c >= U'a' && c <= U'z') || c == U'_' ||
               (c >= 0xc0 && c <= 0x2af) || (c >= 0x370 && c <= 0x52f) ||
               (c >= 0x3040 && c <= 0x30ff) || (c >= 0x3400 && c <= 0x9fff) ||
               (c >= 0xac00 && c <= 0xd7af) || (c >= 0x20000 && c <= 0x323af);
    });
}
std::vector<std::u32string> split(std::u32string_view s, std::u32string_view marks,
                                 bool keep, bool decimal = false) {
    std::vector<std::u32string> out;
    size_t start = 0;
    for (size_t i = 0; i < s.size(); ++i) {
        if (!contains(marks, s[i]) ||
            (decimal && s[i] == U'.' && i && i + 1 < s.size() &&
             digit(s[i - 1]) && digit(s[i + 1])))
            continue;
        out.emplace_back(s.substr(start, i - start + (keep ? 1 : 0)));
        start = i + 1;
    }
    if (start < s.size()) out.emplace_back(s.substr(start));
    return out;
}
void replace(std::u32string &s, std::u32string_view from, std::u32string_view to) {
    size_t at = 0;
    while ((at = s.find(from, at)) != std::u32string::npos) {
        s.replace(at, from.size(), to);
        at += to.size();
    }
}
} // namespace
std::vector<std::string> segment_text(const std::string &input, const std::string &method,
                                    bool english, size_t maximum) {
    if (method.empty()) return {input};
    if (maximum < 1 || method.size() != 4 || method.compare(0, 3, "cut") ||
        method[3] < '0' || method[3] > '5')
        throw std::invalid_argument("invalid text_split_method");
    auto text = decode(input);
    auto first = text.find_first_not_of(U'\n'), last = text.find_last_not_of(U'\n');
    if (first == std::u32string::npos) return {};
    text = text.substr(first, last - first + 1);
    // The upstream frontend adds a lead sentence boundary for short openings.
    auto end = text.find_first_of(boundaries);
    auto opening = text.substr(0, std::min(end, text.size()));
    first = opening.find_first_not_of(U" \t\r\n\v\f");
    last = opening.find_last_not_of(U" \t\r\n\v\f");
    size_t opening_length = first == std::u32string::npos ? 0 : last - first + 1;
    if (!contains(boundaries, text.front()) && opening_length < 4)
        text.insert(text.begin(), english ? U'.' : U'。');
    std::vector<std::u32string> pieces;
    const int mode = method[3] - '0';
    if (mode == 1 || mode == 2) {
        auto marked = text;
        replace(marked, U"……", U"。"); replace(marked, U"——", U"，");
        if (!contains(boundaries, marked.back())) marked += U'。';
        auto sentences = split(marked, boundaries, true);
        if ((mode == 1 && sentences.size() <= 4) || (mode == 2 && sentences.size() < 2))
            pieces.push_back(text);
        else {
            std::u32string current;
            // cut1 keeps the final group with its predecessor, as upstream does.
            size_t groups = (sentences.size() - 1) / 4;
            for (size_t i = 0; i < sentences.size(); ++i) {
                current += sentences[i];
                if ((mode == 1 && (i + 1) % 4 == 0 && (i + 1) / 4 < groups) ||
                    (mode == 2 && current.size() > 50)) {
                    pieces.push_back(std::move(current)); current.clear();
                }
            }
            if (!current.empty()) pieces.push_back(std::move(current));
            if (mode == 2 && pieces.size() > 1 && pieces.back().size() < 50) {
                pieces[pieces.size() - 2] += pieces.back(); pieces.pop_back();
            }
        }
    } else if (mode == 3 || mode == 4) {
        char32_t mark = mode == 3 ? U'。' : U'.';
        first = text.find_first_not_of(mark); last = text.find_last_not_of(mark);
        if (first != std::u32string::npos) {
            text = text.substr(first, last - first + 1);
            // cut4 splits a period only when both neighbours are non-digits.
            if (mode == 3) pieces = split(text, U"。", false);
            else {
                size_t start = 0;
                for (size_t i = 0; i < text.size(); ++i)
                    if (text[i] == U'.' && (!i || !digit(text[i - 1])) &&
                        (i + 1 == text.size() || !digit(text[i + 1]))) {
                        pieces.emplace_back(text.substr(start, i - start)); start = i + 1;
                    }
                pieces.emplace_back(text.substr(start));
            }
        }
    } else if (mode == 5) pieces = split(text, U",.;?!、，。？！：…", true, true);
    else pieces.push_back(text);
    std::vector<std::u32string> joined;
    std::u32string current;
    for (const auto &piece : pieces) {
        auto ignored = mode == 5 ? std::u32string_view(U",.;?!、，。？！：…") : punctuation;
        if (std::all_of(piece.begin(), piece.end(), [&](char32_t c) { return contains(ignored, c); }))
            continue;
        for (auto line : split(piece, U"\n", false))
            if (!line.empty() && line != U" ") {
                current += line;
                if (current.size() >= 5) {
                    joined.push_back(std::move(current)); current.clear();
                }
            }
    }
    if (!current.empty()) {
        if (joined.empty()) joined.push_back(std::move(current));
        else joined.back() += current;
    }
    std::vector<std::string> result;
    for (auto &piece : joined) {
        if (!spoken(piece)) continue;
        if (!contains(boundaries, piece.back())) piece += english ? U'.' : U'。';
        for (size_t start = 0; start < piece.size();) {
            size_t count = std::min(maximum, piece.size() - start);
            if (count < piece.size() - start) {
                auto at = piece.find_last_of(boundaries, start + count - 1);
                if (at != std::u32string::npos && at >= start) count = at - start + 1;
            }
            auto part = piece.substr(start, count);
            if (spoken(part)) result.push_back(encode(part));
            start += count;
        }
    }
    return result;
}
} // namespace tts::text
