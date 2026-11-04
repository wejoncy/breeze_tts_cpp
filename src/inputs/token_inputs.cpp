#include "inputs/token_inputs.h"
#include "core/input_fields.h"
#include "ops/sampling.h"
#include "core/row_capacity.h"
#include <cmath>

namespace tts {
namespace {
void validate_text(const std::string &value, size_t limit, const char *name) {
    if (value.size() > limit || value.find('\0') != std::string::npos)
        throw std::invalid_argument(std::string("invalid ") + name);
}
}
TokenInputs::TokenInputs(const std::string &directory, breeze::ModelConfig config,
                         int batch, int context)
    : tokenizer_(directory + "/tokenizer.json"), config_(std::move(config)), context_(context) {
    definition_.type = config_.backbone.type;
    definition_.reference_program = "reference_codes";
    definition_.codebooks = config_.codebooks;
    definition_.vocabulary = config_.codec.codebook_size;
    definition_.eos = config_.backbone_eos;
    definition_.max_branches = breeze::max_guidance_branches;
    definition_.max_decode_frames = config_.codec.transformer.max_positions;
    const int lanes = batch * definition_.max_branches;
    definition_.programs.push_back({"tokens", dummy(lanes, context), {"hidden_states", "logits", "done"}});
    InputBatch decode;
    decode.limits = {lanes, context, lanes};
    decode.sequence.shape = {lanes, 1, lanes};
    definition_.programs.push_back({"tokens", std::move(decode),
                                   {"hidden_states", "logits", "done"}, ForwardMode::Decode});
    definition_.programs.back().replay = ProgramSpec::Replay::Cache;
    InputBatch audio;
    audio.limits = {batch, 1, batch};
    audio.sequence.shape = {batch, 1, batch};
    audio.bindings = {{"code_rows", TensorView::metadata(batch, 1, Type::I64)},
                       {"histories", {}}};
    audio.parameters = {{"history_stride", double(batch)}};
    definition_.programs.push_back({"waveform", std::move(audio), {"waveform", "pcm"}});
    definition_.programs.back().replay = ProgramSpec::Replay::Batch;
    definition_.programs.back().history = "audio";
    InputBatch reference;
    const int samples = config_.max_reference_frames * config_.samples_per_frame;
    reference.limits = {1, samples, samples};
    reference.sequence.shape = {1, samples, samples};
    reference.uploads.push_back({"waveform", own_tensor(reference, {samples, 1}, Type::F32)});
    definition_.programs.push_back({"reference_codes", std::move(reference), {"codes"}});
}
InputBatch TokenInputs::dummy(int batch, int time) const {
    InputBatch result;
    result.sequence = own_sequence(result, std::vector<int>(batch, time));
    result.sequences.push_back({"text", own_sequence(result, std::vector<int>(2 * batch, time))});
    result.limits = {batch, time, 2 * batch * time};
    result.uploads.push_back({"tokens", own_tensor(result, {2 * batch * time, 1}, Type::I32)});
    result.bindings.push_back({"segments", TensorView::metadata(0, 3, Type::I32)});
    return result;
}
GenerationSegment TokenInputs::encode(const Input &request, const GenerationOptions &options,
                                       size_t index) const {
    validate_text(request.text, 65536, "text");
    validate_text(request.instruction, 16384, "instruction");
    validate_text(request.speaker, 256, "speaker");
    if (request.text.empty()) throw std::invalid_argument("text is required");
    const int branches = options.dual_cfg ? 3 : options.cfg_scale != 1 ? 2 : 1;
    if (branches == 2 && request.instruction.empty()) throw std::invalid_argument("CFG requires instruction");
    if (options.dual_cfg && (!request.reference || request.instruction.empty()))
        throw std::invalid_argument("dual CFG requires reference and instruction");
    std::string speaker = request.speaker;
    if (!speaker.empty() && !(speaker.front() == '[' && speaker.back() == ']')) speaker = "[" + speaker + "]";
    GenerationSegment segment;
    auto &input = segment.input;
    std::vector<int> tokens, segments, text_lengths;
    int destination = 0;
    auto append = [&](const std::vector<int> &values, int output) {
        if (values.empty()) return;
        segments.insert(segments.end(), {int(tokens.size()), int(values.size()), output});
        tokens.insert(tokens.end(), values.begin(), values.end());
        text_lengths.push_back(int(values.size()));
    };
    for (int branch = 0; branch < branches; ++branch) {
        const bool instruction = !request.instruction.empty() && (branches == 3 ? branch == 2 : branch == 0);
        const bool reference = request.reference && (branches != 3 || branch == 1);
        const int first = destination;
        if (reference) {
            const auto &source = *request.reference;
            validate_text(source.text, 65536, "reference transcript");
            if (source.text.empty() || source.codes.empty() || source.codebooks != config_.codebooks ||
                source.codebook_size != config_.codec.codebook_size || source.codes.size() % config_.codebooks ||
                std::any_of(source.codes.begin(), source.codes.end(),
                    [&](int value) { return value < 0 || value >= config_.codec.codebook_size; }))
                throw std::invalid_argument("reference requires transcript and complete audio frames");
            const auto encoded = tokenizer_.encode(speaker + source.text);
            append(encoded, destination);
            destination += int(encoded.size());
            auto codes = source.codes;
            codes.resize(codes.size() + config_.codebooks, config_.codebook_eos);
            auto values = own_integers(input, codes, config_.codebooks);
            input.regions.push_back({values, destination});
            destination += values.rows;
        }
        auto encoded = tokenizer_.encode(speaker + (instruction
            ? config_.instruction_bos + request.instruction + config_.instruction_eos : "") + request.text);
        append(encoded, destination);
        destination += int(encoded.size());
        const int length = destination - first;
        if (length + options.max_frames > context_) throw std::invalid_argument("prompt plus output exceeds context");
        segment.lengths.push_back(length);
        segment.capacities.push_back(length + options.max_frames);
    }
    if (std::any_of(tokens.begin(), tokens.end(), [&](int value) { return value < 0 || value >= config_.text_vocab; }))
        throw std::invalid_argument("tokenizer ID exceeds the text vocabulary");
    input.uploads.push_back({"tokens", own_integers(input, tokens)});
    input.bindings.push_back({"segments", own_integers(input, segments, 3)});
    input.offsets = {{"segments", "text", 0}, {"segments", {}, 2}};
    input.sequence = own_sequence(input, segment.lengths);
    input.sequences.push_back({"text", own_sequence(input, text_lengths)});
    input.limits = {branches, input.sequence.shape.time, row_capacity(destination)};
    segment.max_tokens = options.max_frames;
    segment.seed = request_seed(options.seed, index);
    return segment;
}
std::unique_ptr<RequestPreparation> TokenInputs::begin(const std::vector<Input> &requests,
                                                       const GenerationOptions &options) const {
    auto valid = [&](float temperature, int count, float probability) {
        return std::isfinite(temperature) && temperature >= 0 && count >= 0 && count <= definition_.vocabulary &&
               std::isfinite(probability) && probability > 0 && probability <= 1;
    };
    if (requests.empty() || options.max_frames < 1 || options.max_frames > config_.codec.transformer.max_positions ||
        !valid(options.temperature, options.top_k, options.top_p) ||
        !valid(options.depth_temperature, options.depth_top_k, options.depth_top_p) ||
        !std::isfinite(options.cfg_scale) || options.cfg_scale <= 0 ||
        !std::isfinite(options.cfg_scale_ref) || !std::isfinite(options.cfg_scale_ins) ||
        !std::isfinite(options.repetition_penalty) || options.repetition_penalty <= 0)
        throw std::invalid_argument("invalid generation options");
    if (options.speed_factor != 1 || options.sample_steps || options.cfg_rate != -1)
        throw std::invalid_argument("token speech inputs do not define acoustic speed or diffusion controls");
    auto result = std::make_unique<RequestPlan>();
    result->options = options;
    for (size_t index = 0; index < requests.size(); ++index) {
        if (!requests[index].text_split_method.empty() || requests[index].fragment_interval != 0)
            throw std::invalid_argument("token speech inputs do not define text segmentation controls");
        GenerationInput input;
        input.segments.push_back(encode(requests[index], options, index));
        result->inputs.push_back(std::move(input));
    }
    return std::make_unique<ImmediatePreparation>(std::move(result));
}
std::unique_ptr<ReferencePlan> TokenInputs::prepare_reference(const std::vector<float> &mono,
    std::string transcript, const PreparationExecutor &, const std::string &) const {
    validate_text(transcript, 65536, "reference transcript");
    if (transcript.empty() || mono.empty() || mono.size() > size_t(config_.max_reference_frames) * config_.samples_per_frame)
        throw std::invalid_argument("invalid reference transcript or audio length");
    auto result = std::make_unique<ReferencePlan>();
    result->transcript = std::move(transcript);
    result->program = "reference_codes";
    result->input.limits = {1, int(mono.size()), int(mono.size())};
    result->input.sequence.shape = {1, int(mono.size()), int(mono.size())};
    TensorView source(const_cast<float *>(mono.data()), int(mono.size()), 1, Type::F32);
    auto storage = std::make_shared<Tensor>(source);
    result->input.uploads.push_back({"waveform", storage->view()});
    result->input.owners.push_back(std::move(storage));
    result->outputs = {"codes"};
    return result;
}
std::shared_ptr<const Reference> TokenInputs::finish_reference(ReferencePlan &plan, HostResults outputs) const {
    const auto tensor = outputs.at("codes").view();
    const auto *values = tensor.ptr<int>();
    return std::make_shared<const Reference>(std::move(plan.transcript),
        std::vector<int>(values, values + tensor.numel()), config_.codebooks, config_.codec.codebook_size);
}
} // namespace tts
