// Adapted from Open JTalk: Copyright (c) 2008-2016 Nagoya Institute of Technology.
// BSD-3-Clause; see licenses/linguistics.txt for the full notice and terms.
#include "text/japanese_grammar.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace tts::text::japanese {
Rules::Rules(const std::string &path) : data(read_json(path)), max_mora(data.at("max_mora")) {
    if (max_mora < 1) throw std::invalid_argument("invalid Japanese prosody limit");
    for (const auto &[name, value] : data.at("constants").items())
        constants.emplace(name, value.get_ref<const std::string &>().c_str());
    for (const auto &[name, values] : data.at("tables").items()) {
        if (!values.is_array() || values.empty() || !values.back().is_null())
            throw std::invalid_argument("Japanese rule table must have an end marker");
        auto &table = tables[name];
        for (const auto &value : values)
            table.push_back(value.is_null() ? nullptr : value.get_ref<const std::string &>().c_str());
    }
}
const char *Rules::constant(std::string_view name) const {
    auto found = constants.find(name);
    if (found == constants.end()) throw std::invalid_argument("missing Japanese rule: " + std::string(name));
    return found->second;
}
const std::vector<const char *> &Rules::table(std::string_view name) const {
    auto found = tables.find(name);
    if (found == tables.end()) throw std::invalid_argument("missing Japanese table: " + std::string(name));
    return found->second;
}
namespace {
std::vector<std::string> split(std::string_view text, char delimiter) {
    std::vector<std::string> result;
    for (size_t cursor = 0;;) {
        auto end = text.find(delimiter, cursor);
        result.emplace_back(text.substr(cursor, end == text.npos ? text.size() - cursor : end - cursor));
        if (end == text.npos) return result;
        cursor = end + 1;
    }
}
} // namespace
Node *Document::create() { storage.emplace_back(); return &storage.back(); }
void Document::load(Node *node, std::string_view feature) {
    auto fields = split(feature, ','); fields.resize(std::max(fields.size(), size_t(13)));
    node->pos = fields[1]; node->pos_group1 = fields[2];
    node->pos_group2 = fields[3]; node->pos_group3 = fields[4];
    node->ctype = fields[5]; node->cform = fields[6]; node->chain_rule = fields[11];
    if (fields[12] == "1" || fields[12] == "0") node->chain_flag = fields[12][0] - '0';
    const auto &accent = fields[10];
    auto count = size_t(std::count(accent.begin(), accent.end(), '/'));
    if (accent.find('*') != accent.npos || !count) {
        node->string = fields[0]; node->orig = fields[7];
        node->read = fields[8]; node->pron = fields[9];
        node->acc = node->mora_size = 0;
        return;
    }
    auto accents = split(accent, ':');
    auto originals = split(fields[7], ':'), readings = split(fields[8], ':'), pronunciations = split(fields[9], ':');
    if (accents.size() != count || originals.size() < count || readings.size() < count || pronunciations.size() < count)
        throw std::invalid_argument("Japanese chained dictionary feature alignment");
    size_t cursor = 0;
    for (size_t i = 0; i < count; ++i) {
        if (i) {
            auto *previous = node;
            node = create(); *node = *previous;
            node->chain_flag = 0; node->prev = previous; node->next = nullptr;
            previous->next = node;
        }
        node->string = i + 1 < count ? originals[i] : fields[0].substr(cursor);
        cursor += originals[i].size();
        node->orig = originals[i]; node->read = readings[i]; node->pron = pronunciations[i];
        auto slash = accents[i].find('/');
        node->acc = std::max(0, std::atoi(accents[i].c_str()));
        node->mora_size = std::max(0, std::atoi(accents[i].c_str() + slash + 1));
    }
}
void Document::append(std::string_view feature) {
    auto *node = create(); load(node, feature);
    if (!head) head = node;
    else { tail->next = node; node->prev = tail; }
    while (node->next) node = node->next;
    tail = node;
}
Node *Document::remove(Node *node) {
    auto *next = node->next;
    if (node->prev) node->prev->next = next; else head = next;
    if (next) next->prev = node->prev; else tail = node->prev;
    return next;
}
void Document::remove_silent() {
    for (auto *node = head; node;)
        if (node->pron.empty() || node->pron == "*") node = remove(node);
        else node = node->next;
}
std::vector<std::string> prosody(const Document &document, const Rules &rules) {
    struct Mora { std::vector<std::string> phones; size_t phrase; int position; };
    struct Phrase { int accent, moras = 0; };
    std::vector<Mora> moras;
    std::vector<Phrase> phrases;
    bool pause = false;
    const std::string short_pause = rules.constant("JPCOMMON_MORA_SHORT_PAUSE"),
                      question = rules.constant("JPCOMMON_MORA_QUESTION"),
                      long_vowel = rules.constant("JPCOMMON_MORA_LONG_VOWEL"),
                      unvoice = rules.constant("JPCOMMON_MORA_UNVOICE");
    const auto &mapping = rules.table("jpcommon_mora_list");
    for (auto *node = document.head; node; node = node->next) {
        if (node->pron == short_pause || node->pron == question) { pause = true; continue; }
        bool first = true;
        std::string_view pronunciation(node->pron);
        while (!pronunciation.empty()) {
            if (pronunciation.substr(0, long_vowel.size()) == long_vowel) {
                if (!moras.empty() && !pause) {
                    const auto &previous = moras.back();
                    if (!previous.phones.empty()) {
                        auto phrase = previous.phrase;
                        moras.push_back({{previous.phones.back()}, phrase, ++phrases[phrase].moras});
                    }
                }
                pronunciation.remove_prefix(long_vowel.size());
                continue;
            }
            if (pronunciation.substr(0, unvoice.size()) == unvoice) {
                // The model's prosody frontend drops the unvoiced-vowel flag.
                pronunciation.remove_prefix(unvoice.size()); continue;
            }
            size_t i = 0;
            for (; mapping.at(i); i += 3)
                if (pronunciation.substr(0, std::strlen(mapping[i])) == mapping[i]) break;
            if (!mapping.at(i)) break; // Same unknown-pronunciation truncation as JPCommon.
            if (first) {
                if (pause && !moras.empty()) moras.push_back({{"pau"}, size_t(-1), 0});
                pause = false;
                if (phrases.empty() || node->chain_flag != 1) phrases.push_back({node->acc});
                first = false;
            }
            auto phrase = phrases.size() - 1;
            std::vector<std::string> phones{mapping[i + 1]};
            if (mapping[i + 2]) phones.emplace_back(mapping[i + 2]);
            moras.push_back({std::move(phones), phrase, ++phrases[phrase].moras});
            pronunciation.remove_prefix(std::strlen(mapping[i]));
        }
    }
    std::vector<std::string> output;
    auto limit = [&](int value) { return std::clamp(value, 1, rules.max_mora); };
    for (size_t i = 0; i < moras.size(); ++i) {
        const auto &mora = moras[i];
        if (mora.phrase == size_t(-1)) { output.emplace_back("_"); continue; }
        const auto &phrase = phrases[mora.phrase];
        const int accent = phrase.accent ? phrase.accent : phrase.moras;
        const int a1 = std::clamp(mora.position - accent, -rules.max_mora, rules.max_mora), a2 = limit(mora.position),
                  a3 = limit(phrase.moras - mora.position + 1), f1 = limit(phrase.moras);
        for (size_t p = 0; p < mora.phones.size(); ++p) {
            auto phone = mora.phones[p];
            if (phone.size() == 1 && std::string("AEIOU").find(phone[0]) != std::string::npos) phone[0] += 'a' - 'A';
            output.push_back(phone);
            int next = p + 1 < mora.phones.size() ? a2 : i + 1 < moras.size() && moras[i + 1].phrase != size_t(-1) ? limit(moras[i + 1].position) : -50;
            if (a3 == 1 && next == 1 && std::string("aeiouAEIOUNcl").find(phone) != std::string::npos) output.emplace_back("#");
            else if (a1 == 0 && next == a2 + 1 && a2 != f1) output.emplace_back("]");
            else if (a2 == 1 && next == 2) output.emplace_back("[");
        }
    }
    return output;
}
} // namespace tts::text::japanese
