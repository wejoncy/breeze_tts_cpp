// BudouX semantic boundary model, native adaptation (Apache-2.0, Google LLC).
#include "text/budoux.h"
#include "text/utf8.h"
#include <unordered_map>
#include <cstdint>

namespace tts::text {
struct Budoux::Impl {
    // Each feature has fixed length <= 3; Unicode scalars use 21 bits,
    // so the full feature key fits in 63 bits without hashing the text.
    static uint64_t key(std::u32string_view text) {
        uint64_t value = 0;
        for (auto cp : text) {
            if (cp > 0x10ffff) return UINT64_MAX;
            value = (value << 21) | uint32_t(cp);
        }
        return value;
    }
    struct Feature {
        int offset, length;
        std::unordered_map<uint64_t, int> weights;
    };
    std::vector<Feature> features;
    int64_t base = 0;
    explicit Impl(const Json &model) {
        for (const auto &[name, weights] : model.items()) {
            if (name.size() != 3 || name[1] != 'W' || name[2] < '1' || name[2] > '6')
                throw std::invalid_argument("invalid BudouX feature name");
            int index = name[2] - '0';
            int length = name[0] == 'U' ? 1 : name[0] == 'B' ? 2 : name[0] == 'T' ? 3 : 0;
            if (!length || (length == 2 && index > 3) || (length == 3 && index > 4))
                throw std::invalid_argument("invalid BudouX feature geometry");
            Feature feature{index - (length == 2 ? 3 : 4), length, {}};
            feature.weights.reserve(weights.size());
            for (const auto &[text, weight] : weights.items()) {
                int value = weight;
                auto cp = decode(text);
                if (cp.size() != size_t(length))
                    throw std::invalid_argument("BudouX feature key length mismatch");
                feature.weights.emplace(key(cp), value);
                base -= value;
            }
            features.push_back(std::move(feature));
        }
    }
    std::vector<std::u32string> split(std::u32string_view text) const {
        std::vector<std::u32string> out;
        if (text.empty()) return out;
        size_t first = 0;
        for (size_t i = 1; i < text.size(); ++i) {
            int64_t score = base;
            for (const auto &feature : features) {
                if (int64_t(i) + feature.offset < 0) continue;
                size_t at = size_t(int64_t(i) + feature.offset);
                if (at > text.size() || size_t(feature.length) > text.size() - at) continue;
                auto found = feature.weights.find(key(text.substr(at, feature.length)));
                if (found != feature.weights.end()) score += 2 * int64_t(found->second);
            }
            if (score > 0) { out.emplace_back(text.substr(first, i - first)); first = i; }
        }
        out.emplace_back(text.substr(first));
        return out;
    }
};
Budoux::Budoux(const Json &weights) : impl_(std::make_unique<Impl>(weights)) {}
Budoux::~Budoux() = default;
std::vector<std::u32string> Budoux::split(std::u32string_view text) const { return impl_->split(text); }
} // namespace tts::text
