// Dictionary format and Viterbi semantics follow MeCab (BSD, Taku Kudo and NTT).
// No MeCab library is linked or invoked by the native engine.
#include "text/morphology.h"
#include "text/linguistic_data.h"
#include "text/utf8.h"
#include <algorithm>
#include <limits>

namespace tts::text {
namespace {
struct Lexeme { uint16_t left, right, pos; int16_t cost; std::string_view feature; };
struct Match { size_t bytes; Lexeme lexeme; };
class Dictionary {
    DataReader trie_, tokens_;
    std::string_view features_;
    struct Unit { int32_t base; uint32_t check; };
    Unit unit(size_t index) const {
        if (index >= trie_.remaining() / 8)
            throw std::invalid_argument("morphology trie index");
        auto data = trie_.slice(index * 8);
        return {data.read<int32_t>(), data.read<uint32_t>()};
    }
    Lexeme token(size_t index) const {
        if (index >= tokens_.remaining() / 16)
            throw std::invalid_argument("morphology token index");
        auto data = tokens_.slice(index * 16);
        auto left = data.read<uint16_t>(), right = data.read<uint16_t>();
        auto pos = data.read<uint16_t>();
        auto cost = data.read<int16_t>();
        auto offset = data.read<uint32_t>();
        if (offset >= features_.size())
            throw std::invalid_argument("morphology feature index");
        auto feature = features_.substr(offset);
        auto end = feature.find('\0');
        if (end == std::string_view::npos)
            throw std::invalid_argument("unterminated morphology feature");
        return {left, right, pos, cost, feature.substr(0, end)};
    }
  public:
    Dictionary(LinguisticAsset &asset, const std::string &prefix)
        : trie_(asset.section(prefix + "da")), tokens_(asset.section(prefix + "token")) {
        auto features = asset.section(prefix + "feat");
        features_ = features.text(features.remaining());
        if (trie_.remaining() % 8 || tokens_.remaining() % 16 || trie_.remaining() < 8)
            throw std::invalid_argument("morphology dictionary section sizes");
    }
    std::vector<Match> matches(std::string_view text) const {
        std::vector<Match> out;
        int32_t base = unit(0).base;
        for (size_t i = 0;; ++i) {
            if (base < 0) throw std::invalid_argument("negative morphology trie branch");
            auto terminal = unit(size_t(base));
            if (terminal.check == uint32_t(base) && terminal.base < 0) {
                uint32_t value = uint32_t(-int64_t(terminal.base) - 1);
                for (uint32_t t = 0; t < (value & 255); ++t)
                    out.push_back({i, token((value >> 8) + t)});
            }
            if (i == text.size()) return out;
            auto next = unit(size_t(base) + static_cast<unsigned char>(text[i]) + 1);
            if (next.check != uint32_t(base)) return out;
            base = next.base;
        }
    }
};
struct Character {
    uint32_t bits;
    bool kind(Character other) const { return (bits & other.bits & ((1u << 18) - 1)) != 0; }
    unsigned category() const { return (bits >> 18) & 255; }
    unsigned length() const { return (bits >> 26) & 15; }
    bool group() const { return (bits >> 30) & 1; }
    bool invoke() const { return bits >> 31; }
};
} // namespace
struct Morphology::Impl {
    LinguisticAsset asset;
    Dictionary system, unknown;
    std::vector<Dictionary> users;
    DataReader characters, matrix;
    std::vector<std::vector<Lexeme>> fallback;
    std::vector<std::pair<uint16_t, int32_t>> space_penalties;
    uint32_t left_size, right_size, grouping;
    explicit Impl(const std::string &path)
        : asset(path, "GSVMOR02"), system(asset, "sys"), unknown(asset, "unk"),
          characters(asset.section("charmap")), matrix(asset.section("matrix")) {
        auto meta = asset.section("meta");
        left_size = meta.read<uint32_t>();
        right_size = meta.read<uint32_t>();
        grouping = meta.read<uint32_t>();
        auto count = meta.read<uint32_t>();
        for (uint32_t i = 0; i < count; ++i) users.emplace_back(asset, "u" + std::to_string(i));
        meta.finish();
        if (!left_size || !right_size || !grouping ||
            uint64_t(left_size) * right_size * 2 != matrix.remaining() ||
            characters.remaining() % 4 || characters.remaining() < 128 * 4)
            throw std::invalid_argument("morphology matrix/character geometry");
        auto names = asset.section("names");
        count = names.read<uint32_t>();
        if (count > 18 || count > names.remaining())
            throw std::invalid_argument("morphology character category count");
        while (count--) {
            auto name = names.text(names.read<uint8_t>());
            std::vector<Lexeme> tokens;
            for (const auto &match : unknown.matches(name))
                if (match.bytes == name.size()) tokens.push_back(match.lexeme);
            if (tokens.empty()) throw std::invalid_argument("missing morphology unknown category");
            fallback.push_back(std::move(tokens));
        }
        names.finish();
        auto penalties = asset.section("penalty");
        count = penalties.read<uint32_t>();
        if (count > penalties.remaining() / 6)
            throw std::invalid_argument("morphology space penalty count");
        while (count--) {
            auto pos = penalties.read<uint16_t>();
            auto cost = penalties.read<int32_t>();
            space_penalties.emplace_back(pos, cost);
        }
        penalties.finish();
    }
    Character character(char32_t cp) const {
        if (cp >= characters.remaining() / 4) cp = 0;
        return {characters.slice(size_t(cp) * 4).read<uint32_t>()};
    }
    int transition(uint16_t right, uint16_t left) const {
        if (right >= left_size || left >= right_size)
            throw std::invalid_argument("morphology connection ID outside matrix");
        return matrix.slice((size_t(right) + size_t(left_size) * left) * 2).read<int16_t>();
    }
    std::vector<Morphology::Token> analyze(std::string_view input) const {
        std::vector<size_t> offsets;
        auto text = decode(input, &offsets);
        const size_t count = text.size(), absent = std::numeric_limits<size_t>::max();
        struct Node { size_t first, end, parent; Lexeme word; int64_t cost; };
        std::vector<Node> nodes{{0, 0, absent, {0, 0, 0, 0, {}}, 0}};
        std::vector<std::vector<size_t>> ends(count + 1);
        ends[0].push_back(0);
        const auto whitespace = character(U' ');
        for (size_t position = 0; position < count; ++position) {
            if (ends[position].empty()) continue;
            size_t begin = position;
            Character previous = whitespace;
            while (begin < count && previous.kind(character(text[begin])))
                previous = character(text[begin++]);
            if (begin == count) continue;
            auto info = character(text[begin]);
            std::vector<std::pair<size_t, Lexeme>> candidates;
            auto remaining = input.substr(offsets[begin],
                std::min(input.size() - offsets[begin], size_t(UINT16_MAX)));
            auto candidates_from = [&](const Dictionary &dictionary) {
                for (const auto &match : dictionary.matches(remaining)) {
                    auto end = std::lower_bound(offsets.begin() + begin, offsets.end(),
                                                offsets[begin] + match.bytes);
                    if (end == offsets.end() || *end != offsets[begin] + match.bytes || !match.bytes)
                        throw std::invalid_argument("morphology token splits UTF-8 character");
                    candidates.emplace_back(size_t(end - offsets.begin()), match.lexeme);
                }
            };
            candidates_from(system);
            for (const auto &dictionary : users) candidates_from(dictionary);
            auto unknown_word = [&](size_t end) {
                if (info.category() >= fallback.size())
                    throw std::invalid_argument("morphology unknown category index");
                for (const auto &word : fallback[info.category()])
                    candidates.emplace_back(end, word);
            };
            if (candidates.empty() || info.invoke()) {
                size_t group_end = absent;
                if (info.group()) {
                    size_t end = begin + 1;
                    previous = info;
                    while (end < count && previous.kind(character(text[end])))
                        previous = character(text[end++]);
                    if (end - begin - 1 <= grouping) unknown_word(end);
                    group_end = end;
                }
                size_t end = begin + 1;
                for (unsigned length = 1; length <= info.length() && end <= count; ++length) {
                    if (end == group_end) continue;
                    unknown_word(end);
                    if (end == count || !info.kind(character(text[end]))) break;
                    ++end;
                }
                if (candidates.empty()) unknown_word(end);
            }
            // MeCab prepends dictionary candidates and connected nodes. Preserve
            // both orders so equal-cost paths select the same token sequence.
            for (auto item = candidates.rbegin(); item != candidates.rend(); ++item) {
                int64_t best = std::numeric_limits<int64_t>::max();
                size_t parent = absent;
                int penalty = 0;
                if (begin != position)
                    for (auto entry : space_penalties)
                        if (entry.first == item->second.pos) { penalty = entry.second; break; }
                for (auto before = ends[position].rbegin(); before != ends[position].rend(); ++before) {
                    const auto &node = nodes[*before];
                    int64_t cost = node.cost + transition(node.word.right, item->second.left) +
                                   item->second.cost + penalty;
                    if (cost < best) { best = cost; parent = *before; }
                }
                nodes.push_back({begin, item->first, parent, item->second, best});
                ends[item->first].push_back(nodes.size() - 1);
            }
        }
        size_t end = count;
        while (ends[end].empty()) --end;
        int64_t best = std::numeric_limits<int64_t>::max();
        size_t selected = absent;
        for (auto it = ends[end].rbegin(); it != ends[end].rend(); ++it) {
            auto cost = nodes[*it].cost + transition(nodes[*it].word.right, 0);
            if (cost < best) { best = cost; selected = *it; }
        }
        std::vector<Morphology::Token> out;
        for (; selected && selected != absent; selected = nodes[selected].parent) {
            const auto &node = nodes[selected];
            out.push_back({std::string(input.substr(offsets[node.first],
                                                  offsets[node.end] - offsets[node.first])),
                           std::string(node.word.feature)});
        }
        std::reverse(out.begin(), out.end());
        return out;
    }
};
Morphology::Morphology(const std::string &path) : impl_(std::make_unique<Impl>(path)) {}
Morphology::~Morphology() = default;
std::vector<Morphology::Token> Morphology::analyze(std::string_view input) const {
    return impl_->analyze(input);
}
} // namespace tts::text
