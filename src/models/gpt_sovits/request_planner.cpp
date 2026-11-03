#include "models/gpt_sovits/request_planner.h"
#include "io/audio_io.h"
#include "text/segmentation.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstring>

namespace tts::gpt_sovits {
Planner::Planner(const std::string &directory, const Json &c, int context, Type type)
    : frontend_(directory, c.at("bert").at("max_position_embeddings")), context_(context),
      output_rate_(c.at("sample_rate")), samples_per_frame_(c.at("samples_per_frame")),
      reference_rate_(c.at("reference").at("sampling_rate")),
      hubert_rate_(c.at("reference").at("hubert_sample_rate")),
      rate_(c.at("acoustic").at("semantic_repeat"), c.at("acoustic").at("feature_repeat")),
      min_seconds_(c.at("reference").at("min_seconds")),
      max_seconds_(c.at("reference").at("max_seconds")),
      silence_seconds_(c.at("reference").at("semantic_silence_seconds")), type_(type),
      sampling_{c.at("flow").at("steps"), c.at("flow").at("cfg_scale")},
      windows_{c.at("synthesis").at("window_frames"), c.at("synthesis").at("target_frames"),
               c.at("synthesis").at("stream_core_frames"),
               c.at("synthesis").at("stream_future_frames"),
               c.at("synthesis").at("stream_holdback_frames"),
               c.at("synthesis").at("vocoder_context_frames"),
               c.at("synthesis").at("stream_overlap_frames"),
               double(output_rate_) / samples_per_frame_} {}
class Planner::Preparation final : public RequestPreparation {
    const Planner &owner_;
    std::vector<Input> inputs_;
    std::unique_ptr<Plan> plan_ = std::make_unique<Plan>();
    std::unique_ptr<Frontend::Preparation> text_;
    std::shared_ptr<const ReferenceData> reference_;
    std::vector<std::string> segments_;
    std::vector<TextInput> prepared_;
    SynthesisWindows windows_;
    size_t segment_ = 0;
    size_t current_ = 0;
  public:
    Preparation(const Planner &owner, const std::vector<Input> &inputs,
                 const GenerationOptions &options)
        : owner_(owner), inputs_(inputs) {
        plan_->options = options;
    }
    std::unique_ptr<PreparationTask> next() override {
        while (current_ < inputs_.size()) {
            const auto &input = inputs_[current_];
            if (!text_) {
                if (!input.instruction.empty())
                    throw std::invalid_argument(
                        "GPT-SoVITS conditions on reference speech, not instructions");
                if (!std::isfinite(input.fragment_interval) || input.fragment_interval < 0 ||
                    input.fragment_interval > maximum_fragment_interval)
                    throw std::invalid_argument("fragment_interval must be between 0 and 10 seconds");
                reference_ =
                    input.reference
                        ? std::dynamic_pointer_cast<const ReferenceData>(input.reference->features)
                        : nullptr;
                if (!reference_ || reference_->type != owner_.type_ || reference_->frames < 1 ||
                    reference_->codes.empty() || (!reference_->text.bert_ids.empty() &&
                                                  reference_->bert_projection.empty()))
                    throw std::invalid_argument(
                        "GPT-SoVITS requires a prepared reference with matching dtype");
                if (segments_.empty()) {
                    windows_ = SynthesisWindows::resolve(plan_->options, owner_.windows_,
                                                         reference_->frames);
                    segments_ = text::segment_text(input.text, input.text_split_method,
                                                   input.language == "en", owner_.frontend_.max_characters());
                    if (segments_.empty())
                        throw std::invalid_argument("text has no speakable segments");
                }
                text_ = owner_.frontend_.begin(segments_.at(segment_), input.language);
            }
            if (auto task = text_->next()) return task;
            prepared_.push_back(text_->finish());
            text_.reset();
            if (++segment_ < segments_.size()) continue;
            PlannedInput row;
            row.reference = std::move(reference_);
            row.windows = windows_;
            row.pause_frames = int(std::llround(input.fragment_interval * owner_.output_rate_ /
                                                owner_.samples_per_frame_));
            for (const auto &part : prepared_) {
                if (part.bert_ids.size() > size_t(owner_.context_))
                    throw std::invalid_argument("BERT spans exceed initialized token capacity");
                const int64_t prompt = part.phones.size() + row.reference->text.phones.size() +
                                       row.reference->codes.size();
                if (prompt >= owner_.context_)
                    throw std::invalid_argument("reference and text leave no semantic output context");
                int codes = owner_.rate_.codes_for_frames(plan_->options.max_frames,
                    plan_->options.speed_factor, int(owner_.context_ - prompt));
                row.capacity = std::max(row.capacity, int(prompt) + codes);
                if (&part == &prepared_.front()) row.max_codes = codes;
            }
            row.text = std::move(prepared_.front());
            for (size_t i = 1; i < prepared_.size(); ++i)
                row.segments.push_back(std::move(prepared_[i]));
            plan_->inputs.push_back(std::move(row));
            prepared_.clear(); segments_.clear(); segment_ = 0;
            ++current_;
        }
        return {};
    }
    void accept(std::unique_ptr<PreparationTask> task) override {
        if (!text_) throw std::logic_error("unexpected GPT-SoVITS preparation result");
        text_->accept(std::move(task));
    }
    std::unique_ptr<RequestPlan> finish() override {
        if (current_ != inputs_.size() || text_ || !plan_)
            throw std::logic_error("GPT-SoVITS preparation is incomplete");
        return std::move(plan_);
    }
};
std::unique_ptr<RequestPreparation> Planner::begin(const std::vector<Input> &inputs,
                                                   const GenerationOptions &options) const {
    AcousticRate::validate_speed(options.speed_factor);
    if (inputs.empty() || options.max_frames < 1 ||
        options.max_frames > rate_.frames(context_) || !std::isfinite(options.temperature) ||
        options.temperature < 0 || options.top_k < 0 || !std::isfinite(options.top_p) ||
        options.top_p <= 0 || options.top_p > 1 || !std::isfinite(options.repetition_penalty) ||
        options.repetition_penalty <= 0)
        throw std::invalid_argument("invalid GPT-SoVITS generation limits/sampling options");
    if (options.dual_cfg || options.cfg_scale != 1)
        throw std::invalid_argument(
            "GPT-SoVITS does not use Breeze classifier-free guidance controls");
    auto resolved = options;
    auto sampling = FlowSampling::resolve(options.sample_steps, options.cfg_rate, sampling_);
    resolved.sample_steps = sampling.steps;
    resolved.cfg_rate = sampling.guidance;
    return std::make_unique<Preparation>(*this, inputs, resolved);
}
std::unique_ptr<ReferencePlan> Planner::prepare_reference(const std::vector<float> &mono,
                                                          std::string transcript,
                                                        const PreparationExecutor &execute,
                                                        const std::string &language) const {
    if (mono.size() < size_t(min_seconds_ * output_rate_) ||
        mono.size() > size_t(max_seconds_ * output_rate_) ||
        std::any_of(mono.begin(), mono.end(), [](float x) { return !std::isfinite(x); }))
        throw std::invalid_argument("GPT-SoVITS reference duration/samples outside model limits");
    auto plan = std::make_unique<ReferenceInput>();
    plan->text = frontend_.prepare(transcript, execute, language);
    plan->transcript = std::move(transcript);
    if (plan->text.phones.size() > size_t(context_) || plan->text.bert_ids.size() > size_t(context_))
        throw std::invalid_argument("reference transcript exceeds acoustic context");
    plan->wave = resample_audio(mono.data(), mono.size(), output_rate_, 1,
                                size_t(max_seconds_ * reference_rate_), reference_rate_);
    auto small = resample_audio(mono.data(), mono.size(), output_rate_, 1,
                                size_t(max_seconds_ * hubert_rate_), hubert_rate_);
    // The official checkpoint appends sampling_rate * silence_seconds samples
    // after resampling to HuBERT's rate. Preserve that input convention exactly.
    small.resize(small.size() + size_t(silence_seconds_ * reference_rate_), 0);
    plan->hubert_wave.resize(small.size());
    for (size_t i = 0; i < small.size(); ++i) {
        if (type_ == Type::F16) {
            auto value = __float2half_rn(small[i]);
            std::memcpy(&plan->hubert_wave[i], &value, sizeof(value));
        } else {
            auto value = __float2bfloat16_rn(small[i]);
            std::memcpy(&plan->hubert_wave[i], &value, sizeof(value));
        }
    }
    return plan;
}
} // namespace tts::gpt_sovits
