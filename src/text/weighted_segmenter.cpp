// Word routing and HMM semantics follow Jieba (MIT, Copyright (c) 2013 Sun Junyi).
// Asset format documented by GPT-SoVITS-native (MIT, Copyright (c) 2026 白菜工厂1145号员工).
#include "text/weighted_segmenter.h"
#include "text/linguistic_data.h"
#include "text/utf8.h"
#include "io/json.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <map>

namespace tts::text {
namespace {
constexpr double missing_probability = -3.14e100;
bool digit(char32_t c) { return c >= U'0' && c <= U'9'; }
bool alpha(char32_t c) { return (c >= U'a' && c <= U'z') || (c >= U'A' && c <= U'Z'); }
bool han(char32_t c) { return c >= 0x4e00 && c <= 0x9fd5; }
bool internal(char32_t c) { return han(c) || digit(c) || alpha(c) || std::u32string_view(U"+#&._").find(c) != std::u32string_view::npos; }
struct HMM {
    struct State { char boundary; uint16_t tag; };
    std::vector<State> states;
    std::vector<double> start, transition;
    std::vector<bool> outgoing;
    std::vector<std::vector<std::pair<uint32_t, double>>> emission;
    std::map<char32_t, std::vector<uint16_t>> allowed;
    std::vector<std::vector<uint16_t>> previous;
    std::vector<uint16_t> terminals;
    bool tagged = true;
    double emit(int state, char32_t cp) const {
        const auto &row = emission[state];
        auto it = std::lower_bound(row.begin(), row.end(), uint32_t(cp),
                                  [](auto pair, uint32_t value) { return pair.first < value; });
        return it != row.end() && it->first == cp ? it->second : missing_probability;
    }
    void load(LinguisticAsset &asset, const std::string &start_name,
              const std::string &transition_name, const std::string &emission_name) {
        const size_t count = states.size();
        auto first = asset.section(start_name);
        start.resize(count);
        for (auto &value : start) value = first.read<double>();
        first.finish();
        auto edges = asset.section(transition_name);
        transition.assign(count * count, tagged ? -std::numeric_limits<double>::infinity()
                                               : missing_probability);
        outgoing.resize(count);
        for (size_t from = 0; from < count; ++from) {
            uint32_t n = edges.read<uint32_t>();
            outgoing[from] = n != 0;
            if (n > edges.remaining() / 12) throw std::invalid_argument("HMM edge count");
            while (n--) {
                uint16_t to = edges.read<uint16_t>();
                edges.read<uint16_t>();
                if (to >= count) throw std::invalid_argument("HMM edge state");
                transition[from * count + to] = edges.read<double>();
            }
        }
        edges.finish();
        auto observations = asset.section(emission_name);
        emission.resize(count);
        for (auto &row : emission) {
            auto n = observations.read<uint32_t>();
            if (n > observations.remaining() / 12) throw std::invalid_argument("HMM emission count");
            row.reserve(n);
            while (n--) {
                auto cp = observations.read<uint32_t>();
                auto value = observations.read<double>();
                row.emplace_back(cp, value);
            }
            if (!std::is_sorted(row.begin(), row.end())) throw std::invalid_argument("unordered HMM emissions");
        }
        observations.finish();
    }
    std::vector<uint16_t> path(std::u32string_view word) const {
        if (word.empty()) return {};
        size_t count = states.size();
        if (word.size() > SIZE_MAX / count) throw std::overflow_error("HMM history size");
        std::vector<uint16_t> history(word.size() * count, UINT16_MAX);
        std::vector<double> before(count), after(count);
        std::vector<bool> present(count, !tagged), next(count), reachable(count);
        auto initial = allowed.find(word.front());
        for (size_t s = 0; s < count; ++s) {
            present[s] = !tagged || initial == allowed.end() || initial->second.empty() ||
                std::find(initial->second.begin(), initial->second.end(), s) != initial->second.end();
            before[s] = start[s] + emit(int(s), word.front());
        }
        for (size_t t = 1; t < word.size(); ++t) {
            std::fill(next.begin(), next.end(), !tagged);
            std::fill(reachable.begin(), reachable.end(), false);
            if (tagged) {
                for (size_t from = 0; from < count; ++from) if (present[from] && outgoing[from])
                    for (size_t to = 0; to < count; ++to)
                        if (std::isfinite(transition[from * count + to])) reachable[to] = true;
                if (std::none_of(reachable.begin(), reachable.end(), [](bool x) { return x; }))
                    throw std::runtime_error("HMM has no reachable state");
                auto found = allowed.find(word[t]);
                for (size_t s = 0; s < count; ++s)
                    next[s] = reachable[s] && (found == allowed.end() ||
                        std::find(found->second.begin(), found->second.end(), s) != found->second.end());
                if (std::none_of(next.begin(), next.end(), [](bool x) { return x; }))
                    next = reachable;
                if (std::none_of(next.begin(), next.end(), [](bool x) { return x; }))
                    std::fill(next.begin(), next.end(), true);
            }
            for (size_t to = 0; to < count; ++to) if (next[to]) {
                double best = tagged ? -std::numeric_limits<double>::infinity() : missing_probability;
                uint16_t parent = UINT16_MAX;
                const auto value = emit(int(to), word[t]);
                if (tagged) {
                    for (size_t from = 0; from < count; ++from) if (present[from] && outgoing[from]) {
                        // POS-HMM adds the transition before the emission.
                        auto score = (before[from] + transition[from * count + to]) + value;
                        if (parent == UINT16_MAX || score >= best) { best = score; parent = uint16_t(from); }
                    }
                } else {
                    // finalseg's C extension keeps the first candidate on equal
                    // score. Its exported predecessor order is authoritative.
                    for (auto from : previous[to]) {
                        auto score = (value + before[from]) + transition[size_t(from) * count + to];
                        if (score > best) { best = score; parent = from; }
                    }
                    if (parent == UINT16_MAX)
                        parent = *std::max_element(previous[to].begin(), previous[to].end());
                }
                after[to] = best;
                history[t * count + to] = parent;
            }
            before.swap(after);
            present.swap(next);
        }
        uint16_t last = UINT16_MAX;
        double best = -std::numeric_limits<double>::infinity();
        for (auto s : terminals) if (present[s]) {
            if (last == UINT16_MAX || before[s] > best || (tagged && before[s] == best)) {
                last = s; best = before[s];
            }
        }
        if (last == UINT16_MAX) throw std::runtime_error("HMM has no terminal path");
        std::vector<uint16_t> result(word.size());
        for (size_t t = word.size(); t-- > 0;) {
            result[t] = last;
            if (t) last = history[t * count + last];
            if (t && last >= count) throw std::runtime_error("HMM invalid predecessor");
        }
        return result;
    }
};
} // namespace
struct WeightedSegmenter::Impl {
    struct Entry { uint32_t offset; uint16_t length, tag; int64_t frequency; };
    LinguisticAsset asset;
    std::u32string codepoints;
    std::vector<Entry> entries;
    std::vector<std::string> tags;
    double log_total;
    HMM pos, final;
    Impl(const std::string &directory) : asset(directory + "/jieba.bin", "GSVJTB01") {
        auto meta = asset.section("dictmeta");
        auto total = meta.read<int64_t>();
        if (total <= 0) throw std::invalid_argument("dictionary frequency total");
        log_total = std::log(double(total));
        auto cp = asset.section("cps");
        codepoints.resize(cp.remaining() / 4);
        for (auto &c : codepoints) c = cp.read<uint32_t>();
        cp.finish();
        auto words = asset.section("entries");
        auto n = words.read<uint32_t>();
        if (n > words.remaining() / 16) throw std::invalid_argument("dictionary entry count");
        entries.resize(n);
        for (auto &e : entries) {
            e.offset = words.read<uint32_t>(); e.length = words.read<uint16_t>();
            e.tag = words.read<uint16_t>(); e.frequency = words.read<int64_t>();
            if (!e.length || e.offset > codepoints.size() || e.length > codepoints.size() - e.offset || e.frequency < 0)
                throw std::invalid_argument("dictionary entry bounds");
        }
        words.finish();
        auto labels = asset.section("tags");
        n = labels.read<uint32_t>();
        if (n > labels.remaining()) throw std::invalid_argument("dictionary tag count");
        while (n--) tags.emplace_back(labels.text(labels.read<uint8_t>()));
        labels.finish();
        auto states = asset.section("states");
        n = states.read<uint32_t>();
        if (!n || n >= UINT16_MAX || n > states.remaining() / 4) throw std::invalid_argument("HMM state count");
        pos.states.resize(n);
        for (auto &s : pos.states) {
            s.boundary = char(states.read<uint8_t>()); s.tag = states.read<uint16_t>(); states.read<uint8_t>();
            if (s.tag >= tags.size()) throw std::invalid_argument("HMM tag index");
        }
        states.finish();
        pos.terminals.resize(n); std::iota(pos.terminals.begin(), pos.terminals.end(), uint16_t(0));
        pos.load(asset, "hmmstart", "hmmtrans", "hmmitmap");
        auto candidates = asset.section("chars");
        struct Character { uint32_t cp, offset, count; };
        n = candidates.read<uint32_t>();
        if (n > candidates.remaining() / 12) throw std::invalid_argument("HMM character count");
        std::vector<Character> chars(n);
        for (auto &c : chars) { c.cp = candidates.read<uint32_t>(); c.offset = candidates.read<uint32_t>(); c.count = candidates.read<uint32_t>(); }
        n = candidates.read<uint32_t>();
        if (n > candidates.remaining() / 2) throw std::invalid_argument("HMM candidate count");
        std::vector<uint16_t> pool(n);
        for (auto &s : pool) { s = candidates.read<uint16_t>(); if (s >= pos.states.size()) throw std::invalid_argument("HMM candidate state"); }
        candidates.finish();
        for (auto c : chars) {
            if (c.offset > pool.size() || c.count > pool.size() - c.offset) throw std::invalid_argument("HMM candidate range");
            pos.allowed[c.cp] = std::vector<uint16_t>(pool.begin() + c.offset, pool.begin() + c.offset + c.count);
        }
        auto config = read_json(directory + "/linguistics.json").at("finalseg");
        auto boundaries = config.at("states").get<std::string>();
        final.tagged = false;
        for (auto c : boundaries) final.states.push_back({c, 0});
        final.previous = config.at("previous").get<std::vector<std::vector<uint16_t>>>();
        final.terminals = config.at("terminals").get<std::vector<uint16_t>>();
        if (final.previous.size() != final.states.size() || final.terminals.empty()) throw std::invalid_argument("finalseg policy dimensions");
        for (const auto &row : final.previous) {
            if (row.empty()) throw std::invalid_argument("empty finalseg predecessors");
            for (auto s : row) if (s >= final.states.size()) throw std::invalid_argument("finalseg predecessor state");
        }
        for (auto s : final.terminals) if (s >= final.states.size()) throw std::invalid_argument("finalseg terminal state");
        final.load(asset, "fstart", "ftrans", "femit");
    }
    std::u32string_view key(const Entry &e) const { return {codepoints.data() + e.offset, e.length}; }
    const Entry *find(std::u32string_view word) const {
        auto it = std::lower_bound(entries.begin(), entries.end(), word,
                                  [&](const Entry &e, auto w) { return key(e) < w; });
        return it != entries.end() && key(*it) == word ? &*it : nullptr;
    }
    void emit(std::u32string_view word, std::vector<SegToken> &out, std::string tag = {}) const {
        if (tag.empty()) { auto *e = find(word); tag = e && e->tag < tags.size() ? tags[e->tag] : "x"; }
        out.push_back({std::u32string(word), std::move(tag)});
    }
    void hmm(std::u32string_view word, bool tagged, std::vector<SegToken> &out) const {
        const auto &model = tagged ? pos : final;
        auto route = model.path(word);
        size_t begin = 0, next = 0;
        for (size_t i = 0; i < route.size(); ++i) {
            auto state = model.states[route[i]];
            if (state.boundary == 'B') begin = i;
            else if (state.boundary == 'E' || state.boundary == 'S') {
                auto first = state.boundary == 'S' ? i : begin;
                emit(word.substr(first, i + 1 - first), out, tagged ? tags[state.tag] : "x");
                next = i + 1;
            }
        }
        if (next < word.size()) emit(word.substr(next), out, tagged ? tags[model.states[route[next]].tag] : "x");
    }
    void detail(std::u32string_view word, bool tagged, std::vector<SegToken> &out) const {
        for (size_t first = 0; first < word.size();) {
            size_t end = first + 1;
            bool chinese = han(word[first]);
            if (chinese) {
                while (end < word.size() && han(word[end])) ++end;
                hmm(word.substr(first, end - first), tagged, out);
            } else {
                bool number = digit(word[first]) || word[first] == '.';
                bool latin = alpha(word[first]) || digit(word[first]);
                while (end < word.size() && !han(word[end]) &&
                    (tagged ? (number ? digit(word[end]) || word[end] == '.' :
                        latin ? alpha(word[end]) || digit(word[end]) : !alpha(word[end]) && !digit(word[end]) && word[end] != '.') :
                        (latin ? alpha(word[end]) || digit(word[end]) : !alpha(word[end]) && !digit(word[end])))) ++end;
                if (!tagged && latin) {
                    if (end + 1 < word.size() && word[end] == '.' && digit(word[end + 1])) {
                        ++end; while (end < word.size() && digit(word[end])) ++end;
                    }
                    if (end < word.size() && word[end] == '%') ++end;
                }
                emit(word.substr(first, end - first), out, tagged ? (number ? "m" : latin ? "eng" : "x") : "x");
            }
            first = end;
        }
    }
    void dag(std::u32string_view word, bool tagged, std::vector<SegToken> &out) const {
        struct Route { double score = 0; size_t end = 0; };
        std::vector<Route> route(word.size() + 1);
        for (size_t first = word.size(); first-- > 0;) {
            bool found = false;
            for (size_t end = first + 1; end <= word.size(); ++end) {
                auto *entry = find(word.substr(first, end - first));
                if (!entry) break;
                if (entry->frequency) {
                    auto score = (std::log(double(entry->frequency)) - log_total) + route[end].score;
                    if (!found || score >= route[first].score) { route[first] = {score, end}; found = true; }
                }
            }
            if (!found) route[first] = {-log_total + route[first + 1].score, first + 1};
        }
        size_t buffered = 0, first = 0;
        auto flush = [&](size_t end) {
            if (buffered == end) return;
            auto part = word.substr(buffered, end - buffered);
            auto *entry = find(part);
            if (part.size() == 1) emit(part, out);
            else if (!entry || !entry->frequency) detail(part, tagged, out);
            else for (char32_t c : part) emit(std::u32string_view(&c, 1), out);
        };
        while (first < word.size()) {
            auto end = route[first].end;
            if (end > first + 1) { flush(first); emit(word.substr(first, end - first), out); buffered = end; }
            first = end;
        }
        flush(first);
    }
    std::vector<SegToken> cut(std::u32string_view word, bool tagged) const {
        std::vector<SegToken> out;
        for (size_t first = 0; first < word.size();) {
            bool block = internal(word[first]) || (!tagged && word[first] == '%');
            size_t end = first + 1;
            if (block) {
                while (end < word.size() && (internal(word[end]) || (!tagged && word[end] == '%'))) ++end;
                dag(word.substr(first, end - first), tagged, out);
            } else {
                if (word[first] == U'\r' && end < word.size() && word[end] == U'\n') ++end;
                emit(word.substr(first, end - first), out, "x");
            }
            first = end;
        }
        return out;
    }
};
WeightedSegmenter::WeightedSegmenter(const std::string &directory) : impl_(std::make_unique<Impl>(directory)) {}
WeightedSegmenter::~WeightedSegmenter() = default;
std::vector<SegToken> WeightedSegmenter::segment(std::u32string_view word) const { return impl_->cut(word, true); }
std::vector<std::u32string> WeightedSegmenter::search(std::u32string_view word) const {
    std::vector<std::u32string> out;
    for (const auto &token : impl_->cut(word, false)) {
        for (size_t length : {size_t(2), size_t(3)}) if (token.word.size() > length)
            for (size_t first = 0; first + length <= token.word.size(); ++first) {
                auto part = std::u32string_view(token.word).substr(first, length);
                auto *entry = impl_->find(part);
                if (entry && entry->frequency) out.emplace_back(part);
            }
        out.push_back(token.word);
    }
    return out;
}
} // namespace tts::text
