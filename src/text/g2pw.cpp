#include "text/g2pw.h"
#include "core/input_fields.h"
#include "text/token_alignment.h"
#include "text/utf8.h"
#include "io/json.h"
#include <algorithm>
#include <unordered_map>

namespace tts::text {
namespace {
using Dictionary = std::unordered_map<std::u32string, std::u32string>;
Dictionary dictionary(const Json &json) {
    Dictionary result;
    for (const auto &[key, value] : json.items())
        result.emplace(decode(key), decode(value.get<std::string>()));
    return result;
}
std::u32string replace(const std::u32string &source, const Dictionary &words, size_t max_length) {
    std::u32string result;
    for (size_t i = 0; i < source.size();) {
        size_t length = std::min(max_length, source.size() - i);
        auto found = words.end();
        while (length && (found = words.find(source.substr(i, length))) == words.end())
            --length;
        if (length) {
            result += found->second;
            i += length;
        } else
            result += source[i++];
    }
    return result;
}
std::string numbered(const std::string &marked, const Dictionary &marks) {
    auto word = decode(marked);
    std::u32string plain;
    char32_t tone = U'5';
    for (size_t i = 0; i < word.size();) {
        size_t length = std::min<size_t>(2, word.size() - i);
        auto found = marks.end();
        while (length && (found = marks.find(word.substr(i, length))) == marks.end())
            --length;
        if (!length) {
            plain += word[i++];
            continue;
        }
        auto value = found->second;
        if (!value.empty() && value.back() >= U'1' && value.back() <= U'5') {
            tone = value.back();
            value.pop_back();
        }
        plain += value;
        i += length;
    }
    plain += tone;
    return encode(plain);
}
} // namespace
struct G2PW::Impl {
    const Tokenizer &tokenizer;
    std::vector<std::string> labels;
    std::unordered_map<char32_t, int> characters;
    std::unordered_map<char32_t, std::string> monophonic;
    Dictionary traditional, variants, tone_marks;
    size_t max_phrase = 1;
    int context, max_tokens;
    Impl(const std::string &directory, const Tokenizer &t) : tokenizer(t) {
        auto data = read_json(directory + "/g2pw.json");
        labels = data.at("labels").get<std::vector<std::string>>();
        context = data.at("context_chars");
        max_tokens = data.at("max_tokens");
        if (labels.empty() || context < 0 || max_tokens < 3)
            throw std::invalid_argument("invalid G2PW metadata");
        auto chars = data.at("characters").get<std::vector<std::string>>();
        auto allowed = data.at("allowed").get<std::vector<std::vector<int>>>();
        if (allowed.size() != chars.size())
            throw std::invalid_argument("G2PW character mask count");
        for (size_t i = 0; i < chars.size(); ++i) {
            auto cp = decode(chars[i]);
            if (cp.size() != 1)
                throw std::invalid_argument("invalid G2PW character");
            characters.emplace(cp[0], int(i));
        }
        for (auto &character : data.at("exclude_polyphonic"))
            characters.erase(decode(character.get<std::string>()).at(0));
        for (const auto &[key, value] : data.at("monophonic").items())
            monophonic.emplace(decode(key).at(0), value.get<std::string>());
        traditional = dictionary(data.at("traditional_characters"));
        for (auto &[key, value] : dictionary(data.at("traditional_phrases")))
            traditional[key] = std::move(value);
        variants = dictionary(data.at("taiwan_variants"));
        tone_marks = dictionary(data.at("tone_marks"));
        for (const auto &[key, value] : traditional)
            max_phrase = std::max(max_phrase, key.size());

    }
    std::unique_ptr<G2PWTask> plan(const std::u32string &sentence,
                                    std::vector<std::string> &result,
                                    std::vector<size_t> &queries) const {
        for (size_t i = 0; i < sentence.size(); ++i) {
            auto mono = monophonic.find(sentence[i]);
            if (characters.count(sentence[i]))
                queries.push_back(i);
            else if (mono != monophonic.end() && !mono->second.empty())
                result[i] = mono->second;
        }
        if (queries.empty())
            return {};
        size_t left =
            context ? (queries.front() > size_t(context) ? queries.front() - context : 0) : 0;
        size_t right = context ? std::min(sentence.size(), queries.back() + size_t(context) + 1)
                               : sentence.size();
        auto input = encode(std::u32string_view(sentence).substr(left, right - left));
        std::vector<size_t> bytes;
        decode(input, &bytes);
        auto tokens = tokenizer.tokenize(input);
        if (tokens.size() > size_t(max_tokens))
            throw std::invalid_argument("G2PW text exceeds BERT context");
        auto task = std::make_unique<G2PWTask>();
        task->ids.reserve(tokens.size());
        for (const auto &token : tokens)
            task->ids.push_back(int(token.id));
        TokenSpanCursor alignment(tokens);
        for (auto query : queries) {
            task->characters.push_back(characters.at(sentence[query]));
            size_t byte = bytes.at(query - left);
            auto found = alignment.find(byte);
            if (found == tokens.size())
                throw std::invalid_argument("G2PW query has no tokenizer span");
            task->positions.push_back(int(found));
        }
        task->offsets.push_back(int(task->ids.size()));
        return task;
    }
};
G2PW::G2PW(const std::string &directory, const Tokenizer &tokenizer,
           const PypinyinResolver &fallback)
    : impl_(std::make_unique<Impl>(directory, tokenizer)), fallback_(fallback) {}
G2PW::~G2PW() = default;
G2PW::Resolver::Resolver(const G2PW &owner, const std::vector<std::u32string> &segments)
    : owner_(owner) {
    const auto &impl = *owner_.impl_;
    readings_.reserve(segments.size());
    Batch block;
    auto flush = [&] {
        if (block.targets.empty()) return;
        batches_.push_back(std::move(block));
        block = Batch{};
    };
    for (size_t index = 0; index < segments.size(); ++index) {
        const auto &original = segments[index];
        auto sentence = replace(replace(original, impl.traditional, impl.max_phrase),
                                impl.variants, 1);
        if (sentence.size() != original.size())
            throw std::invalid_argument("G2PW traditional conversion changed text length");
        auto simplified = sentence;
        for (auto &cp : simplified) cp = owner_.fallback_.t2s(cp);
        std::vector<PinyinSyl> fallback;
        owner_.fallback_.resolve(simplified.data(), simplified.size(), &fallback);
        readings_.emplace_back(original, std::vector<std::string>{});
        auto &result = readings_.back().second;
        result.reserve(original.size());
        for (const auto &syllable : fallback) {
            if (!syllable.raw)
                result.push_back(numbered(syllable.text, impl.tone_marks));
            else
                for (auto cp : decode(syllable.text))
                    result.push_back(encode(std::u32string_view(&cp, 1)));
        }
        if (result.size() != original.size())
            throw std::invalid_argument("G2PW fallback alignment changed");
        std::vector<size_t> queries;
        auto plan = impl.plan(sentence, result, queries);
        if (!plan) continue;
        if (block.task->ids.size() + plan->ids.size() > size_t(impl.max_tokens)) flush();
        int offset = int(block.task->ids.size());
        block.task->ids.insert(block.task->ids.end(), plan->ids.begin(), plan->ids.end());
        block.task->offsets.push_back(int(block.task->ids.size()));
        for (size_t i = 0; i < queries.size(); ++i) {
            block.task->positions.push_back(offset + plan->positions[i]);
            block.task->characters.push_back(plan->characters[i]);
            block.targets.emplace_back(index, queries[i]);
        }
    }
    flush();
}
std::unique_ptr<PreparationTask> G2PW::Resolver::next() {
    if (pending_) throw std::logic_error("pronunciation task is still pending");
    if (current_ == batches_.size()) return {};
    pending_ = true;
    return std::move(batches_[current_].task);
}
void G2PW::Resolver::accept(std::unique_ptr<PreparationTask> completed) {
    if (!pending_ || !completed)
        throw std::logic_error("unexpected pronunciation task result");
    const auto &output = dynamic_cast<const G2PWTask &>(*completed);
    const auto &targets = batches_[current_].targets;
    if (output.labels.numel() != targets.size())
        throw std::runtime_error("pronunciation query count mismatch");
    for (size_t i = 0; i < targets.size(); ++i) {
        const auto &label = owner_.impl_->labels.at(output.labels.data_ptr<int>()[i]);
        if (!label.empty()) readings_[targets[i].first].second[targets[i].second] = label;
    }
    ++current_;
    pending_ = false;
}
bool G2PW::Resolver::resolveSentenceTone3(const char32_t *word, size_t count,
                                        std::vector<std::string> *out) const {
    std::u32string_view sentence(word, count);
    if (pending_ || current_ != batches_.size())
        throw std::logic_error("pronunciation preparation is incomplete");
    auto found = std::find_if(readings_.begin(), readings_.end(), [&](const auto &reading) {
        return std::u32string_view(reading.first) == sentence;
    });
    if (found == readings_.end())
        throw std::logic_error("sentence is absent from pronunciation preparation");
    *out = found->second;
    return true;
}
} // namespace tts::text

namespace tts::text {
InputBatch G2PWTask::input() const {
    if (ids.empty() || positions.empty() || positions.size() != characters.size() ||
        offsets.size() < 2 || offsets.front() || offsets.back() != int(ids.size()))
        throw std::invalid_argument("invalid pronunciation task dimensions");
    InputBatch result;
    std::vector<int> lengths;
    for (size_t sequence = 1; sequence < offsets.size(); ++sequence) {
        const int length = offsets[sequence] - offsets[sequence - 1];
        if (length < 1) throw std::invalid_argument("invalid pronunciation sequence extent");
        lengths.push_back(length);
    }
    for (int position : positions)
        if (position < 0 || size_t(position) >= ids.size())
            throw std::invalid_argument("pronunciation query outside its token sequence");
    result.sequence = own_sequence(result, lengths);
    result.limits = result.sequence.shape;
    auto tensor = [](const std::vector<int> &values) {
        return TensorView(const_cast<int *>(values.data()), int(values.size()), 1, Type::I32);
    };
    result.uploads = {{"input_ids", tensor(ids)}, {"positions", tensor(positions)},
                      {"characters", tensor(characters)}};
    result.offsets = {{"positions", {}, 0}};
    return result;
}
void G2PWTask::accept(HostResults outputs) {
    const auto tensor = outputs.at("labels").view();
    if (tensor.device != -1 || tensor.type != Type::I32 || tensor.numel() != positions.size())
        throw std::invalid_argument("pronunciation output shape/type mismatch");
    labels = std::move(outputs.at("labels"));
}
} // namespace tts::text
