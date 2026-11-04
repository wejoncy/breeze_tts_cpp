#include "inputs/phoneme_inputs.h"
#include "core/input_fields.h"
#include "core/row_capacity.h"
#include "text/frontend.h"
#include "text/segmentation.h"
#include "io/audio_io.h"
#include "io/model_loader.h"
#include "models/gpt_sovits/sampling.h"
#include "ops/storage.h"
#include <numeric>

namespace tts {
namespace {
struct PhonemeReference final : ReferenceFeatures {
    text::TextInput text;
    HostResults tensors;
    Type type;
    int frames = 0;
};
struct PhonemeReferencePlan final : ReferencePlan { text::TextInput text; };
class PhonemeInputs final : public RequestPlanner {
    Json config_;
    text::TextFrontend frontend_;
    int context_, output_rate_, reference_rate_, speech_rate_, reference_frames_, hidden_;
    Type type_;
    SequenceRate rate_;
    WindowLimits limits_;
    InputDefinition definition_;
    class Preparation;
    InputBatch fields(const text::TextInput &text, const PhonemeReference &reference) const;
    GenerationInput encode(const Input &, std::vector<text::TextInput>, std::shared_ptr<const PhonemeReference>,
                            const GenerationOptions &, size_t index) const;

  public:
    PhonemeInputs(const ModelLoader &loader, int batch, int context, Type type)
        : config_(loader.config()), frontend_(loader.path(""), config_.at("bert").at("max_position_embeddings")),
          context_(context), output_rate_(config_.at("sample_rate")),
          reference_rate_(config_.at("reference").at("sampling_rate")),
          speech_rate_(config_.at("reference").at("hubert_sample_rate")),
          reference_frames_(config_.at("synthesis").at("reference_frames")),
          hidden_(config_.at("semantic").at("hidden_dim")), type_(type),
          rate_(config_.at("acoustic").at("semantic_repeat"), config_.at("acoustic").at("feature_repeat")),
          limits_{config_.at("synthesis").at("window_frames"), config_.at("synthesis").at("target_frames"),
              config_.at("synthesis").at("stream_core_frames"), config_.at("synthesis").at("stream_future_frames"),
              config_.at("synthesis").at("stream_holdback_frames"), config_.at("synthesis").at("vocoder_context_frames"),
              config_.at("synthesis").at("stream_overlap_frames"),
              double(output_rate_) / config_.at("samples_per_frame").get<int>()} {
        definition_.type = type;
        definition_.vocabulary = definition_.eos = config_.at("semantic").at("EOS");
        definition_.min_tokens = config_.at("synthesis").at("min_semantic_tokens");
        definition_.eos_on_argmax = definition_.nucleus_first = true;
        definition_.include_threshold = false;
        definition_.query_heads = config_.at("semantic").at("head");
        definition_.head_dimensions = hidden_ / definition_.query_heads;
        definition_.max_audio_frames = limits_.target + std::max(limits_.left,
            int(std::ceil(SynthesisWindows::maximum_overlap_seconds * limits_.frames_per_second)));
        text::TextInput dummy_text;
        dummy_text.phones = {0}; dummy_text.phone_bert_rows = {0};
        dummy_text.bert_ids = {0, 0, 0}; dummy_text.bert_offsets = {0, 3};
        PhonemeReference dummy_reference;
        dummy_reference.text = dummy_text; dummy_reference.type = type;
        dummy_reference.tensors["codes"] = Tensor::zeros({1, 1}, Type::I32);
        auto input = fields(dummy_text, dummy_reference);
        input.limits = {batch, context, batch * context};
        definition_.programs.push_back({"tokens", std::move(input), {"logits", "done"}});
        InputBatch decode;
        decode.limits = {batch, 1, batch}; decode.sequence.shape = {batch, 1, batch};
        definition_.programs.push_back({"tokens", std::move(decode), {"logits", "done"},
                                       ForwardMode::Decode, ProgramSpec::Replay::Cache});
        InputBatch audio;
        const int time = limits_.left + limits_.core + limits_.future;
        audio.limits = {batch, definition_.max_audio_frames, context};
        audio.sequence.shape = {batch, time, batch * time};
        const int minimum_reference = int(std::min(double(reference_frames_),
            config_.at("reference").at("min_seconds").get<double>() *
            config_.at("reference").at("mel_sample_rate").get<int>() /
            config_.at("reference").at("mel_hop").get<int>()));
        ProgramSpec waveform{"waveform", std::move(audio), {"waveform", "audio_layout"},
                              ForwardMode::Prefill, ProgramSpec::Replay::Regions};
        waveform.regions = {
            {"flow", {reference_frames_ + limits_.core + limits_.future,
                       reference_frames_ + limits_.core,
                       minimum_reference + limits_.core + limits_.future,
                       minimum_reference + limits_.core},
                std::min(config_.at("synthesis").at("window_frames").get<int>(),
                         reference_frames_ + definition_.max_audio_frames)},
            {"vocoder", {limits_.core + limits_.future, limits_.left + limits_.core + limits_.future,
                          limits_.left + limits_.core}, definition_.max_audio_frames}};
        definition_.programs.push_back(std::move(waveform));
        InputBatch auxiliary;
        const int tokens = read_json(loader.path("g2pw.json")).at("max_tokens");
        if (tokens < 1 || batch > INT32_MAX / tokens)
            throw std::invalid_argument("pronunciation batch capacity exceeds tensor dimensions");
        const int rows = batch * tokens;
        auxiliary.limits = {rows, tokens, rows};
        for (const char *name : {"input_ids", "positions", "characters"})
            auxiliary.uploads.push_back({name, own_tensor(auxiliary, {rows, 1}, Type::I32)});
        // Positive sequence lengths bound the sequence count by total rows.
        auxiliary.sequence = own_sequence(auxiliary, std::vector<int>(rows, 1));
        definition_.programs.push_back({"pronunciation", std::move(auxiliary), {"labels"}});
        InputBatch reference;
        reference.limits = {1, context, context};
        reference.sequence.shape = {1, 1, 1};
        const auto &settings = config_.at("reference");
        const float maximum_seconds = settings.at("max_seconds");
        const int waveform_samples = int(maximum_seconds * reference_rate_);
        const int speech_samples = int(maximum_seconds * speech_rate_ +
            settings.at("semantic_silence_seconds").get<float>() * reference_rate_);
        reference.uploads = {{"waveform", own_tensor(reference, {waveform_samples, 1}, Type::F32)},
            {"speech_waveform", own_tensor(reference, {speech_samples, 1}, type)},
            {"phones", own_tensor(reference, {context, 1}, Type::I32)},
            {"input_ids", own_tensor(reference, {3, 1}, Type::I32)}};
        reference.sequences.push_back({"text", own_sequence(reference, {3})});
        definition_.programs.push_back({"reference", std::move(reference),
            {"codes", "style", "mel", "features", "waveform_condition", "reference_bert"}});
    }
    const InputDefinition &definition() const override { return definition_; }
    std::unique_ptr<RequestPreparation> begin(const std::vector<Input> &, const GenerationOptions &) const override;
    std::unique_ptr<ReferencePlan> prepare_reference(const std::vector<float> &, std::string,
        const PreparationExecutor &, const std::string &) const override;
    std::shared_ptr<const Reference> finish_reference(ReferencePlan &, HostResults) const override;
};
InputBatch PhonemeInputs::fields(const text::TextInput &text, const PhonemeReference &reference) const {
    InputBatch input;
    auto upload = [&](const char *name, const std::vector<int> &values, int columns = 1) {
        input.uploads.push_back({name, own_integers(input, values, columns)});
    };
    const auto codes = reference.tensors.at("codes").view();
    const int phones = int(reference.text.phones.size() + text.phones.size()), prompt = phones + codes.rows;
    std::vector<int> phone_ids = reference.text.phones;
    phone_ids.insert(phone_ids.end(), text.phones.begin(), text.phones.end());
    std::vector<int> phone_rows(phones), code_rows(codes.rows), code_positions(codes.rows);
    std::iota(phone_rows.begin(), phone_rows.end(), 0);
    std::iota(code_rows.begin(), code_rows.end(), phones);
    std::iota(code_positions.begin(), code_positions.end(), 0);
    std::vector<int> feature_rows(reference.text.phones.size(), -1);
    feature_rows.insert(feature_rows.end(), text.phone_bert_rows.begin(), text.phone_bert_rows.end());
    InputAddress address{"reference_features", "reference_bert", {}};
    for (int row : reference.text.phone_bert_rows)
        address.offsets.push_back(row < 0 ? -1 : int64_t(row) * hidden_ * bytes(type_));
    address.offsets.resize(phones, -1);
    input.addresses.push_back(std::move(address));
    upload("input_ids", text.bert_ids);
    upload("phones", phone_ids);
    upload("codes", std::vector<int>(codes.ptr<int>(), codes.ptr<int>() + codes.numel()));
    upload("phone_output_rows", phone_rows);
    upload("phone_positions", phone_rows);
    upload("phone_feature_rows", feature_rows);
    upload("code_output_rows", code_rows);
    upload("code_positions", code_positions);
    upload("last_rows", {prompt - 1});
    upload("prefix_ranges", {0, 0, phones, phones}, 4);
    upload("suffix_ranges", {phones, 0, codes.rows, prompt}, 4);
    input.offsets = {{"phone_output_rows", {}, 0}, {"phone_feature_rows", "text", 0},
        {"code_output_rows", {}, 0}, {"last_rows", {}, 0}, {"prefix_ranges", {}, 0}, {"suffix_ranges", {}, 0}};
    input.batch_bindings = {{"last_rows", BatchField::OutputRows},
        {"prefix_ranges", BatchField::Bidirectional}, {"suffix_ranges", BatchField::Causal}};
    input.sequence = own_sequence(input, {prompt});
    std::vector<int> lengths;
    for (size_t span = 1; span < text.bert_offsets.size(); ++span)
        lengths.push_back(text.bert_offsets[span] - text.bert_offsets[span - 1]);
    input.sequences.push_back({"text", own_sequence(input, lengths)});
    input.limits = {1, prompt, row_capacity(prompt)};
    return input;
}
GenerationInput PhonemeInputs::encode(const Input &request, std::vector<text::TextInput> texts,
    std::shared_ptr<const PhonemeReference> reference, const GenerationOptions &options, size_t index) const {
    GenerationInput result;
    result.rate = rate_;
    result.windows = SynthesisWindows::resolve(options, limits_, reference->frames);
    const int samples_per_frame = config_.at("samples_per_frame");
    result.pause_frames = int(std::llround(request.fragment_interval * output_rate_ / samples_per_frame));
    for (size_t segment_index = 0; segment_index < texts.size(); ++segment_index) {
        const auto &text = texts[segment_index];
        if (text.bert_ids.size() > size_t(context_)) throw std::invalid_argument("BERT spans exceed the input capacity");
        GenerationSegment segment;
        segment.input = fields(text, *reference);
        segment.audio.limits = {1, 1, 1};
        segment.audio.sequence.shape = {1, 1, 1};
        auto speed = own_tensor(segment.audio, {1, 1}, Type::F64);
        auto guidance = own_tensor(segment.audio, {1, 1}, Type::F32);
        *speed.ptr<double>() = options.speed_factor;
        *guidance.ptr<float>() = options.cfg_rate;
        segment.audio.bindings = {
            {"phone_lengths", own_integers(segment.audio, {int(text.phones.size())})},
            {"reference_frames", own_integers(segment.audio, {reference->frames})},
            {"steps", own_integers(segment.audio, {options.sample_steps})},
            {"retain_features", own_integers(segment.audio, {int(result.windows.mode == SynthesisMode::Hybrid)})},
            {"defer_audio", own_integers(segment.audio, {int(result.windows.mode == SynthesisMode::Offline)})},
            {"speed", speed}, {"guidance", guidance}};
        segment.input.owners.push_back(reference);
        const int prompt = segment.input.sequence.shape.rows;
        if (prompt >= context_) throw std::invalid_argument("reference and text leave no output context");
        segment.max_tokens = rate_.codes_for_frames(options.max_frames, options.speed_factor, context_ - prompt);
        segment.lengths = {prompt}; segment.capacities = {prompt + segment.max_tokens};
        segment.position_offset = reference->tensors.at("codes").size(0);
        segment.seed = request_seed(request_seed(options.seed, index), segment_index);
        auto phones = own_integers(segment.input, text.phones);
        segment.initializers.push_back({"phones", phones});
        for (const auto &field : reference->tensors) {
            std::string_view name = field.first;
            if (name == "codes") continue;
            if (name == "mel") name = "reference_mel";
            else if (name == "features") name = "reference_features";
            segment.initializers.push_back({name, field.second.view()});
        }
        auto seen = own_tensor(segment.input, {definition_.vocabulary, 1}, Type::I32);
        const auto codes = reference->tensors.at("codes").view();
        for (size_t token = 0; token < codes.numel(); ++token) seen.ptr<int>()[codes.ptr<int>()[token]] = 1;
        segment.initializers.push_back({"seen", seen});
        result.segments.push_back(std::move(segment));
    }
    return result;
}
class PhonemeInputs::Preparation final : public RequestPreparation {
    const PhonemeInputs &owner_;
    std::vector<Input> requests_;
    std::unique_ptr<RequestPlan> plan_ = std::make_unique<RequestPlan>();
    std::unique_ptr<text::TextFrontend::Preparation> pending_;
    std::shared_ptr<const PhonemeReference> reference_;
    std::vector<std::string> segments_;
    std::vector<text::TextInput> texts_;
    size_t input_ = 0, segment_ = 0;
  public:
    Preparation(const PhonemeInputs &owner, const std::vector<Input> &requests, GenerationOptions options)
        : owner_(owner), requests_(requests) { plan_->options = options; }
    std::unique_ptr<PreparationTask> next() override {
        while (input_ < requests_.size()) {
            const auto &request = requests_[input_];
            if (!pending_) {
                if (!request.instruction.empty()) throw std::invalid_argument("reference-conditioned speech does not accept instructions");
                if (!std::isfinite(request.fragment_interval) || request.fragment_interval < 0 || request.fragment_interval > 10)
                    throw std::invalid_argument("fragment_interval must be between 0 and 10 seconds");
                if (!reference_) {
                    reference_ = request.reference ? std::dynamic_pointer_cast<const PhonemeReference>(request.reference->features) : nullptr;
                    if (!reference_ || reference_->type != owner_.type_ || reference_->frames < 1)
                        throw std::invalid_argument("prepared reference with matching dtype is required");
                    segments_ = text::segment_text(request.text, request.text_split_method, request.language == "en", owner_.frontend_.max_characters());
                    if (segments_.empty()) throw std::invalid_argument("text has no speakable segments");
                }
                pending_ = owner_.frontend_.begin(segments_[segment_], request.language);
            }
            if (auto task = pending_->next()) return task;
            texts_.push_back(pending_->finish()); pending_.reset();
            if (++segment_ < segments_.size()) continue;
            plan_->inputs.push_back(owner_.encode(request, std::move(texts_), reference_, plan_->options, input_));
            texts_.clear(); segments_.clear(); reference_.reset(); segment_ = 0; ++input_;
        }
        return {};
    }
    void accept(std::unique_ptr<PreparationTask> task) override { pending_->accept(std::move(task)); }
    std::unique_ptr<RequestPlan> finish() override {
        if (input_ != requests_.size() || pending_) throw std::logic_error("input preparation is incomplete");
        return std::move(plan_);
    }
};
std::unique_ptr<RequestPreparation> PhonemeInputs::begin(const std::vector<Input> &requests,
                                                       const GenerationOptions &options) const {
    SequenceRate::validate_speed(options.speed_factor);
    if (requests.empty() || options.max_frames < 1 || options.max_frames > rate_.frames(context_) ||
        !std::isfinite(options.temperature) || options.temperature < 0 || options.top_k < 0 ||
        !std::isfinite(options.top_p) || options.top_p <= 0 || options.top_p > 1 ||
        !std::isfinite(options.repetition_penalty) || options.repetition_penalty <= 0 || options.dual_cfg || options.cfg_scale != 1)
        throw std::invalid_argument("invalid generation options");
    auto resolved = options;
    const auto sampling = FlowSampling::resolve(options.sample_steps, options.cfg_rate,
        {config_.at("flow").at("steps"), config_.at("flow").at("cfg_scale")});
    resolved.sample_steps = sampling.steps; resolved.cfg_rate = sampling.guidance;
    return std::make_unique<Preparation>(*this, requests, resolved);
}
std::unique_ptr<ReferencePlan> PhonemeInputs::prepare_reference(const std::vector<float> &mono,
    std::string transcript, const PreparationExecutor &execute, const std::string &language) const {
    const auto &settings = config_.at("reference");
    const float minimum = settings.at("min_seconds"), maximum = settings.at("max_seconds");
    if (mono.size() < size_t(minimum * output_rate_) || mono.size() > size_t(maximum * output_rate_) ||
        std::any_of(mono.begin(), mono.end(), [](float value) { return !std::isfinite(value); }))
        throw std::invalid_argument("reference duration or samples outside input limits");
    auto result = std::make_unique<PhonemeReferencePlan>();
    result->text = frontend_.prepare(transcript, execute, language);
    if (result->text.phones.size() > size_t(context_) || result->text.bert_ids.size() > size_t(context_))
        throw std::invalid_argument("reference transcript exceeds context");
    result->transcript = std::move(transcript); result->program = "reference";
    auto &input = result->input;
    input.limits = {1, context_, context_}; input.sequence.shape = {1, 1, 1};
    auto wave = resample_audio(mono.data(), mono.size(), output_rate_, 1, size_t(maximum * reference_rate_), reference_rate_);
    auto speech = resample_audio(mono.data(), mono.size(), output_rate_, 1, size_t(maximum * speech_rate_), speech_rate_);
    speech.resize(speech.size() + size_t(settings.at("semantic_silence_seconds").get<float>() * reference_rate_), 0);
    auto waveform = own_tensor(input, {int(wave.size()), 1}, Type::F32);
    std::copy(wave.begin(), wave.end(), waveform.ptr<float>());
    auto samples = own_tensor(input, {int(speech.size()), 1}, type_);
    ops::convert_host({speech.data(), int(speech.size()), 1, Type::F32}, samples);
    input.uploads = {{"waveform", waveform}, {"speech_waveform", samples},
                      {"phones", own_integers(input, result->text.phones)},
                      {"input_ids", own_integers(input, result->text.bert_ids)}};
    std::vector<int> lengths;
    for (size_t span = 1; span < result->text.bert_offsets.size(); ++span)
        lengths.push_back(result->text.bert_offsets[span] - result->text.bert_offsets[span - 1]);
    input.sequences.push_back({"text", own_sequence(input, lengths)});
    result->outputs = {"codes", "style", "mel", "features", "waveform_condition", "reference_bert"};
    return result;
}
std::shared_ptr<const Reference> PhonemeInputs::finish_reference(ReferencePlan &base, HostResults outputs) const {
    auto &plan = dynamic_cast<PhonemeReferencePlan &>(base);
    auto result = std::make_shared<PhonemeReference>();
    result->text = std::move(plan.text); result->type = type_;
    const int available = std::min(outputs.at("mel").size(0), outputs.at("features").size(0));
    result->frames = std::min(available, reference_frames_);
    for (const char *name : {"mel", "features"}) {
        outputs[name] = outputs.at(name).narrow(0, available - result->frames, result->frames).clone();
    }
    result->tensors = std::move(outputs);
    return std::make_shared<const Reference>(std::move(plan.transcript), std::move(result));
}
} // namespace
std::unique_ptr<RequestPlanner> load_phoneme_inputs(const ModelLoader &loader, int batch, int context, Type type) {
    return std::make_unique<PhonemeInputs>(loader, batch, context, type);
}
} // namespace tts
