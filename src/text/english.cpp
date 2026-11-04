// English rules: GPT-SoVITS (MIT), g2p_en, NLTK and wordsegment (Apache-2.0).
#include "text/english.h"
#include "text/utf8.h"
#include "io/json.h"
#include "models/g2p_en/model.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <regex>
#include <set>
#include <unordered_map>

namespace tts::text {
namespace {
using Strings = std::vector<std::string>;
bool letter(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
bool digit(char c) { return c >= '0' && c <= '9'; }
bool space(char c) { return c == ' ' || (c >= '\t' && c <= '\r'); }
std::string lower(std::string s) {
    for (auto &c : s) if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
    return s;
}
std::string suffix(const std::string &s, size_t count) {
    return s.substr(s.size() > count ? s.size() - count : 0);
}
template <class F> std::string replace(const std::string &s, const std::regex &pattern, F f) {
    std::string out;
    size_t cursor = 0;
    for (auto it = std::sregex_iterator(s.begin(), s.end(), pattern);
         it != std::sregex_iterator(); ++it) {
        out.append(s, cursor, size_t(it->position()) - cursor);
        out += f(*it);
        cursor = size_t(it->position() + it->length());
    }
    out.append(s, cursor, s.size() - cursor);
    return out;
}
void replace_all(std::string &s, const std::string &from, const std::string &to) {
    if (from.empty()) throw std::invalid_argument("empty English replacement");
    size_t p = 0;
    while ((p = s.find(from, p)) != std::string::npos) {
        s.replace(p, from.size(), to);
        p += to.size();
    }
}
} // namespace
struct EnglishPhonemizer::Impl {
    std::unique_ptr<Weights> weights;
    std::unique_ptr<GraphemeToPhoneme> network;
    std::unordered_map<std::string, Strings> dictionary, names;
    struct Homograph { Strings first, second; std::string pos; };
    std::unordered_map<std::string, Homograph> homographs;
    std::unordered_map<std::string, int> symbols, graphemes;
    Strings phonemes, cardinal, cardinal_and, magnitudes;
    std::unordered_map<std::string, std::string> ordinals, years;
    std::vector<std::pair<std::string, std::string>> replacements;
    std::unordered_map<std::string, Strings> measurements;
    Strings measurement_order;
    std::unordered_map<char32_t, std::string> accents;
    std::set<char32_t> combining;
    std::unordered_map<std::string, double> unigrams, bigrams;
    double word_total;
    int word_limit;
    Strings classes;
    std::unordered_map<std::string, std::string> tags;
    std::unordered_map<std::string, std::vector<std::pair<size_t, double>>> pos_weights;
    // The normalized alphabet excludes URLs, markup, emoji and digits. These
    // are the remaining word/punctuation alternatives of TweetTokenizer.
    const std::regex tokens{"[a-z]+(?:[.\\-][a-z]+)*[.][a-z]{2,13}\\b|[A-Za-z](?:[A-Za-z'_-]*[A-Za-z])?|\\S",
                            std::regex::icase};
    const std::regex numbered_item{"\\b([0-9]+)\\. "},
        arithmetic{"(-?(?:[0-9]+(?:\\.[0-9]+)?|\\.[0-9]+|[A-Za-z])(?:⁰|¹|²|³|⁴|⁵|⁶|⁷|⁸|⁹|ˣ|ʸ|ⁿ)*)\\s+(\\+|-|×|÷|=)\\s+(-?(?:[0-9]+(?:\\.[0-9]+)?|\\.[0-9]+|[A-Za-z])(?:⁰|¹|²|³|⁴|⁵|⁶|⁷|⁸|⁹|ˣ|ʸ|ⁿ)*)"},
        negative{"(^|\\s+)-([0-9]+)"}, comma_number{"[0-9][0-9,]+[0-9]"},
        clock{"\\b([01]?[0-9]|2[0-3]):([0-5][0-9])\\b"},
        pounds_before{"£([0-9.,]*[0-9]+)"}, pounds_after{"([0-9.,]*[0-9]+)£"},
        dollars_before{"\\$([0-9.,]*[0-9]+)"}, dollars_after{"([(0-9.,]*[0-9]+)\\$"},
        decimal{"([0-9]+)\\.\\s*([0-9]+)"}, fraction{"([0-9]+)/([0-9]+)"},
        ordinal_number{"([0-9]+)(st|nd|rd|th)"}, integer{"[0-9]+"},
        ie{"i\\.e\\.", std::regex::icase}, eg{"e\\.g\\.", std::regex::icase};
    std::regex unit_pattern;
    Impl(const std::string &directory) {
        const auto data = read_json(directory + "/english.json");
        dictionary = data.at("dictionary").get<decltype(dictionary)>();
        names = data.at("names").get<decltype(names)>();
        for (auto it = data.at("homographs").begin(); it != data.at("homographs").end(); ++it)
            homographs.emplace(it.key(), Homograph{it.value().at(0).get<Strings>(),
                it.value().at(1).get<Strings>(), it.value().at(2).get<std::string>()});
        auto index = [](const Json &rows) {
            std::unordered_map<std::string, int> out;
            for (const auto &row : rows)
                if (!out.emplace(row.get<std::string>(), int(out.size())).second)
                    throw std::invalid_argument("duplicate English alphabet entry");
            return out;
        };
        symbols = index(data.at("symbols")); graphemes = index(data.at("graphemes"));
        phonemes = data.at("phonemes").get<Strings>();
        cardinal = data.at("cardinal").get<Strings>();
        cardinal_and = data.at("cardinal_and").get<Strings>();
        magnitudes = data.at("magnitudes").get<Strings>();
        ordinals = data.at("ordinals").get<decltype(ordinals)>();
        years = data.at("years").get<decltype(years)>();
        measurements = data.at("measurement").get<decltype(measurements)>();
        measurement_order = data.at("measurement_order").get<Strings>();
        std::string pattern = "\\b([0-9]+(\\.[0-9]+)?)(";
        for (const auto &unit : measurement_order) {
            if (pattern.back() != '(') pattern += '|';
            pattern += unit;
        }
        unit_pattern = std::regex(pattern + ")\\b");
        for (const auto &r : data.at("replacements"))
            replacements.emplace_back(r.at(0).get<std::string>(), r.at(1).get<std::string>());
        for (auto it = data.at("accents").begin(); it != data.at("accents").end(); ++it)
            accents.emplace(char32_t(std::stoul(it.key())), it.value().get<std::string>());
        combining = data.at("combining").get<decltype(combining)>();
        unigrams = data.at("unigrams").get<decltype(unigrams)>();
        bigrams = data.at("bigrams").get<decltype(bigrams)>();
        word_total = data.at("word_total"); word_limit = data.at("word_limit");
        const auto &pos = data.at("pos");
        classes = pos.at("classes").get<Strings>();
        std::sort(classes.begin(), classes.end());
        tags = pos.at("tagdict").get<decltype(tags)>();
        for (auto f = pos.at("weights").begin(); f != pos.at("weights").end(); ++f) {
            auto &values = pos_weights[f.key()];
            for (auto t = f.value().begin(); t != f.value().end(); ++t) {
                auto found = std::lower_bound(classes.begin(), classes.end(), t.key());
                if (found == classes.end() || *found != t.key())
                    throw std::invalid_argument("unknown English POS class");
                values.emplace_back(size_t(found - classes.begin()), t.value().get<double>());
            }
        }
        if (cardinal.size() != 1000 || cardinal_and.size() != cardinal.size() ||
            classes.empty() || !(word_total > 0) || word_limit < 1)
            throw std::invalid_argument("invalid English linguistic data");
        ModelLoader::Checkpoint checkpoint;
        checkpoint.tensors.open_file(directory + "/english.safetensors");
        weights = std::make_unique<Weights>(Device::cpu(), Quantization::None, std::move(checkpoint));
        network = std::make_unique<GraphemeToPhoneme>(*weights, data);
        weights->load(); weights->seal();
    }
    std::string number(std::string s, bool conjunction = false) const {
        size_t first = s.find_first_not_of('0');
        if (first == std::string::npos) return cardinal[0];
        s.erase(0, first);
        const auto &table = conjunction ? cardinal_and : cardinal;
        std::string out;
        for (size_t start = 0; start < s.size();) {
            size_t digits = (s.size() - start - 1) % 3 + 1;
            int n = std::stoi(s.substr(start, digits));
            size_t rank = (s.size() - start - digits) / 3;
            if (n) {
                if (!out.empty()) out += !rank && n < 100 ? conjunction ? " and " : " " : ", ";
                out += table.at(n);
                if (rank) {
                    if (rank > magnitudes.size()) throw std::invalid_argument("English number exceeds asset range");
                    out += " " + magnitudes[rank - 1];
                }
            }
            start += digits;
        }
        return out;
    }
    std::string ordinal(std::string words) const {
        size_t p = words.find_last_of(" -"); p = p == std::string::npos ? 0 : p + 1;
        auto found = ordinals.find(words.substr(p));
        if (found == ordinals.end()) throw std::invalid_argument("missing English ordinal word");
        words.replace(p, words.size() - p, found->second);
        return words;
    }
    Strings pos(const Strings &words) const {
        auto normalized = [](const std::string &word) {
            if (word.find('-') != std::string::npos && word[0] != '-') return std::string("!HYPHEN");
            if (word.size() == 4 && std::all_of(word.begin(), word.end(), digit)) return std::string("!YEAR");
            if (!word.empty() && digit(word[0])) return std::string("!DIGITS");
            return lower(word);
        };
        Strings context{"-START-", "-START2-"}, result;
        for (const auto &w : words) context.push_back(normalized(w));
        context.insert(context.end(), {"-END-", "-END2-"});
        std::string prev = "-START-", prev2 = "-START2-";
        for (size_t n = 0; n < words.size(); ++n) {
            auto known = tags.find(words[n]);
            std::string tag;
            if (known != tags.end()) tag = known->second;
            else {
                size_t i = n + 2;
                Strings features{"bias", "i suffix " + suffix(words[n], 3), "i pref1 " + words[n].substr(0, 1),
                    "i-1 tag " + prev, "i-2 tag " + prev2, "i tag+i-2 tag " + prev + " " + prev2,
                    "i word " + context[i], "i-1 tag+i word " + prev + " " + context[i],
                    "i-1 word " + context[i-1], "i-1 suffix " + suffix(context[i-1], 3),
                    "i-2 word " + context[i-2], "i+1 word " + context[i+1],
                    "i+1 suffix " + suffix(context[i+1], 3), "i+2 word " + context[i+2]};
                std::vector<double> scores(classes.size());
                for (const auto &f : features) {
                    auto found = pos_weights.find(f);
                    if (found != pos_weights.end())
                        for (auto [label, weight] : found->second) scores[label] += weight;
                }
                size_t best = 0;
                for (size_t c = 1; c < scores.size(); ++c) if (scores[c] >= scores[best]) best = c;
                tag = classes[best];
            }
            result.push_back(tag); prev2 = std::move(prev); prev = std::move(tag);
        }
        return result;
    }
    Strings predict(const std::string &word) const {
        std::vector<int> ids;
        ids.reserve(word.size());
        for (char character : word) {
            auto found = graphemes.find(std::string(1, character));
            ids.push_back(found == graphemes.end() ? graphemes.at("<unk>") : found->second);
        }
        int offsets[] = {0, int(ids.size())}, count = 0;
        auto output = network->forward(
            {ids.data(), int(ids.size()), 1, Type::I32}, {offsets, 2, 1, Type::I32}, {&count, 1, 1, Type::I32});
        Strings result;
        for (int index = 0; index < count; ++index) result.push_back(phonemes.at(output.data_ptr<int>()[index]));
        return result;
    }
    Strings segment(std::string word) const {
        word.erase(std::remove_if(word.begin(), word.end(), [](char c) { return !letter(c) && !digit(c); }), word.end());
        struct Candidate { double score; Strings words; };
        std::map<std::pair<size_t, std::string>, Candidate> memo;
        std::function<Candidate(size_t, const std::string &)> search = [&](size_t p, const std::string &previous) -> Candidate {
            if (p == word.size()) return {0, {}};
            auto key = std::make_pair(p, previous);
            auto cached = memo.find(key);
            if (cached != memo.end()) return cached->second;
            Candidate best{-INFINITY, {}};
            for (size_t length = 1; length <= std::min<size_t>(word_limit, word.size() - p); ++length) {
                auto prefix = word.substr(p, length);
                auto one = unigrams.find(prefix), two = bigrams.find(previous + " " + prefix), prior = unigrams.find(previous);
                double score = two != bigrams.end() && prior != unigrams.end()
                    ? std::log10(two->second / word_total / (prior->second / word_total))
                    : one != unigrams.end() ? std::log10(one->second / word_total)
                    : std::log10(10. / (word_total * std::pow(10., length)));
                auto candidate = search(p + length, prefix);
                candidate.score += score;
                candidate.words.insert(candidate.words.begin(), std::move(prefix));
                if (candidate.score > best.score || (candidate.score == best.score && candidate.words > best.words))
                    best = std::move(candidate);
            }
            memo.emplace(std::move(key), best);
            return best;
        };
        // wordsegment bounds its recursion with 250-character chunks, carrying
        // the last five words into the next chunk.
        Strings result;
        const std::string remaining = word;
        std::string prefix;
        for (size_t i = 0; i < remaining.size(); i += 250) {
            word = prefix + remaining.substr(i, 250);
            memo.clear();
            auto words = search(0, "<s>").words;
            size_t keep = std::min<size_t>(5, words.size());
            result.insert(result.end(), words.begin(), words.end() - keep);
            prefix.clear();
            for (auto it = words.end() - keep; it != words.end(); ++it) prefix += *it;
        }
        word = prefix;
        memo.clear();
        auto tail = search(0, "<s>").words;
        result.insert(result.end(), tail.begin(), tail.end());
        return result;
    }
    Strings query(const std::string &original) const {
        auto word = lower(original);
        auto known = dictionary.find(word);
        if (word.size() > 1 && known != dictionary.end()) return known->second;
        bool title = !original.empty() && original[0] >= 'A' && original[0] <= 'Z' &&
            std::none_of(original.begin() + 1, original.end(), [](char c) { return c >= 'A' && c <= 'Z'; });
        auto name = names.find(word);
        if (title && name != names.end()) return name->second;
        if (word.size() <= 3) {
            Strings result;
            for (char c : word) {
                auto found = dictionary.find(std::string(1, c));
                if (c == 'a') result.push_back("EY1");
                else if (letter(c) && found != dictionary.end()) result.insert(result.end(), found->second.begin(), found->second.end());
                else result.push_back(std::string(1, c));
            }
            return result;
        }
        if (word.size() > 2 && suffix(word, 2) == "'s" &&
            std::all_of(word.begin(), word.end() - 2, letter)) {
            auto phones = query(word.substr(0, word.size() - 2));
            if (phones.empty()) return phones;
            const auto &last = phones.back();
            if (last == "P" || last == "T" || last == "K" || last == "F" || last == "TH" || last == "HH")
                phones.push_back("S");
            else {
                if (last == "S" || last == "Z" || last == "SH" || last == "ZH" || last == "CH" || last == "JH")
                    phones.push_back("AH0");
                phones.push_back("Z");
            }
            return phones;
        }
        auto parts = segment(word);
        if (parts.size() <= 1) return predict(word);
        Strings phones;
        for (const auto &p : parts) {
            auto next = query(p);
            phones.insert(phones.end(), next.begin(), next.end());
        }
        return phones;
    }
    std::vector<int> pronounce(const std::string &s) const {
        Strings words;
        for (auto it = std::sregex_iterator(s.begin(), s.end(), tokens); it != std::sregex_iterator(); ++it)
            words.push_back(it->str());
        auto labels = pos(words);
        std::vector<int> result;
        for (size_t i = 0; i < words.size(); ++i) {
            auto word = lower(words[i]);
            Strings phones;
            auto homo = homographs.find(word);
            if (std::none_of(word.begin(), word.end(), letter)) phones = {word};
            else if (word.size() == 1) phones = words[i] == "A" ? Strings{"EY1"} : dictionary.at(word);
            else if (homo != homographs.end()) {
                const auto &h = homo->second;
                bool first = labels[i].compare(0, h.pos.size(), h.pos) == 0 ||
                    (labels[i].size() < h.pos.size() && h.pos.compare(0, labels[i].size(), labels[i]) == 0);
                phones = first ? h.first : h.second;
            } else phones = query(words[i]);
            for (auto phone : phones) {
                if (phone == "<unk>") phone = "UNK";
                if (phone == "'") phone = "-";
                auto found = symbols.find(phone);
                if (found != symbols.end()) result.push_back(found->second);
            }
        }
        return result;
    }
    std::string normalize(std::string s) const;
};
std::string EnglishPhonemizer::Impl::normalize(std::string s) const {
    for (const auto &[from, to] : replacements) replace_all(s, from, to);
    // Keep the upstream rule order: ordinals, arithmetic, grouped numbers,
    // times, units, currencies, decimals, fractions, ordinals and cardinals.
    s = replace(s, numbered_item, [](const auto &m) {
        return m[1].str() + "th, "; // The ordinal rule below consumes the suffix.
    });
    while (std::regex_search(s, arithmetic))
        s = replace(s, arithmetic, [](const auto &m) {
            const auto op = m[2].str();
            return m[1].str() + (op == "+" ? " plus " : op == "-" ? " minus " :
                op == "×" ? " times " : op == "÷" ? " divided by " : " Equals ") + m[3].str();
        });
    s = replace(s, negative, [](const auto &m) { return "negative " + m[2].str(); });
    s = replace(s, comma_number, [](const auto &m) {
        auto value = m.str(); value.erase(std::remove(value.begin(), value.end(), ','), value.end()); return value;
    });
    s = replace(s, clock, [&](const auto &m) {
        int h = std::stoi(m[1].str()), minutes = std::stoi(m[2].str());
        return cardinal_and[h > 12 ? h - 12 : h] + (minutes ? " " + cardinal_and[minutes] : " o'clock") +
               (h < 12 ? " a.m." : " p.m.");
    });
    s = replace(s, unit_pattern, [&](const auto &m) {
        const auto value = m[1].str();
        bool singular = !m[2].matched && value.find_first_not_of('0') != std::string::npos &&
                        value.substr(value.find_first_not_of('0')) == "1";
        return value + " " + measurements.at(m[3].str()).at(singular ? 0 : 1);
    });
    auto money = [&](const auto &m, bool pounds) {
        auto value = m[1].str(); auto point = value.find('.');
        const std::string major = pounds ? "pound" : "dollar", minor = pounds ? "penny" : "cent";
        const std::string minors = pounds ? "pence" : "cents";
        if (point != std::string::npos && value.find('.', point + 1) != std::string::npos)
            return value + " " + major + "s";
        auto integral = value.substr(0, point), decimal = point == std::string::npos ? std::string{} : value.substr(point + 1);
        while (decimal.size() < 2) decimal += '0';
        auto strip = [](std::string part) {
            auto first = part.find_first_not_of('0');
            return first == std::string::npos ? std::string("0") : part.substr(first);
        };
        integral = strip(integral); decimal = strip(decimal);
        auto a = integral + " " + major + (integral == "1" ? "" : "s");
        auto b = decimal + " " + (decimal == "1" ? minor : minors);
        return integral != "0" ? a + (decimal != "0" ? " and " + b : "") :
               decimal != "0" ? b : "zero " + major + "s";
    };
    s = replace(s, pounds_before, [&](const auto &m) { return money(m, true); });
    s = replace(s, pounds_after, [&](const auto &m) { return money(m, true); });
    s = replace(s, dollars_before, [&](const auto &m) { return money(m, false); });
    s = replace(s, dollars_after, [&](const auto &m) { return money(m, false); });
    s = replace(s, decimal, [](const auto &m) {
        std::string result = m[1].str() + " point ";
        for (char c : m[2].str()) { if (result.back() != ' ') result += ' '; result += c; }
        return result;
    });
    s = replace(s, fraction, [&](const auto &m) {
        auto a = number(m[1].str(), true), b = number(m[2].str(), true);
        if (b == cardinal[1]) return a;
        bool single = a == cardinal[1];
        bool plural = m[1].str().find_first_not_of('0') != std::string::npos && !single;
        return a + " " + (b == cardinal[2] ? single ? "half" : "halves" : ordinal(b) + (plural ? "s" : ""));
    });
    s = replace(s, ordinal_number, [&](const auto &m) { return ordinal(number(m[1].str(), true)); });
    s = replace(s, integer, [&](const auto &m) {
        auto found = years.find(m.str());
        return found == years.end() ? number(m.str()) : found->second;
    });
    std::string ascii;
    for (char32_t c : decode(s)) {
        auto accent = accents.find(c);
        if (accent != accents.end()) ascii += accent->second;
        else if (!combining.count(c)) ascii += encode(std::u32string(1, c));
    }
    replace_all(ascii, "%", " percent");
    ascii.erase(std::remove_if(ascii.begin(), ascii.end(), [](unsigned char c) {
        return !letter(char(c)) && c != ' ' && c != '\'' && c != '.' && c != ',' &&
               c != '?' && c != '!' && c != '-';
    }), ascii.end());
    ascii = std::regex_replace(ascii, ie, "that is");
    ascii = std::regex_replace(ascii, eg, "for example");
    std::string out;
    for (size_t i = 0; i < ascii.size(); ++i) {
        char c = ascii[i];
        if (c >= 'A' && c <= 'Z' && i && !space(ascii[i-1])) out += ' ';
        out += c;
    }
    const auto punct = [](char c) { return c == '!' || c == '?' || c == ',' || c == '.' || c == '-'; };
    std::string result;
    for (char c : out)
        if (!punct(c) || result.empty() || (!punct(result.back()) && !space(result.back()))) result += c;
    return result;
}
EnglishPhonemizer::EnglishPhonemizer(const std::string &directory)
    : impl_(std::make_unique<Impl>(directory)) {}
EnglishPhonemizer::~EnglishPhonemizer() = default;
std::vector<int> EnglishPhonemizer::pronounce(const std::string &s) const { return impl_->pronounce(s); }
std::string EnglishPhonemizer::normalize(std::string_view s) const { return impl_->normalize(std::string(s)); }
} // namespace tts::text
