#include "execution/prepared_batch.h"
#include <algorithm>
#include <cmath>
namespace tts {
namespace {
void layout(PreparedBatch &);
}
PreparedBatch prepare_batch(const Tokenizer &tokenizer, const ModelConfig &model, int max_batch,
                            int context, const std::vector<Input> &requests,
                            const GenerationOptions &opt) {
    PreparedBatch result;
    int batch = int(requests.size());
    if (batch < 1 || batch > max_batch)
        throw std::invalid_argument("batch size outside configured limit");
    auto valid_sampling = [&](float temperature, int k, float p) {
        return std::isfinite(temperature) && temperature >= 0 && k >= 0 &&
               k <= model.codec.codebook_size && std::isfinite(p) && p > 0 && p <= 1;
    };
    if (opt.max_frames < 1 || opt.max_frames > model.codec.transformer.max_positions ||
        !valid_sampling(opt.temperature, opt.top_k, opt.top_p) ||
        !valid_sampling(opt.depth_temperature, opt.depth_top_k, opt.depth_top_p) ||
        !std::isfinite(opt.cfg_scale) || opt.cfg_scale <= 0 || !std::isfinite(opt.cfg_scale_ref) ||
        !std::isfinite(opt.cfg_scale_ins) || !std::isfinite(opt.repetition_penalty) ||
        opt.repetition_penalty <= 0)
        throw std::runtime_error("invalid generation options");
    int branches = opt.dual_cfg ? 3 : opt.cfg_scale != 1 ? 2 : 1;
    int lanes = batch * branches, maxprompt = 0;
    auto &structured = result.prompts;
    bool &has_reference = result.has_reference;
    auto check_text = [](const std::string &value, size_t limit, const char *name) {
        if (value.size() > limit || value.find('\0') != std::string::npos)
            throw std::invalid_argument(std::string("invalid ") + name);
    };
    for (int branch = 0; branch < branches; branch++)
        for (const auto &req : requests) {
            check_text(req.text, 65536, "text");
            check_text(req.instruction, 16384, "instruction");
            check_text(req.speaker, 256, "speaker");
            if (req.text.empty())
                throw std::invalid_argument("text is required");
            if (branches == 2 && req.instruction.empty())
                throw std::invalid_argument("CFG requires instruction");
            if (opt.dual_cfg && (!req.reference || req.instruction.empty()))
                throw std::invalid_argument("dual CFG requires reference and instruction");
            std::string prefix = req.speaker;
            if (!prefix.empty() && !(prefix.front() == '[' && prefix.back() == ']'))
                prefix = "[" + prefix + "]";
            PreparedPrompt p;
            bool use_instruction =
                !req.instruction.empty() && (branches == 3 ? branch == 2 : branch == 0);
            if (req.reference && (branches != 3 || branch == 1)) {
                check_text(req.reference->text, 65536, "reference transcript");
                if (req.reference->text.empty() || req.reference->codes.empty() ||
                    req.reference->codes.size() % model.codebooks ||
                    req.reference->codebooks != model.codebooks ||
                    req.reference->codebook_size != model.codec.codebook_size)
                    throw std::invalid_argument(
                        "reference requires transcript and complete audio frames");
                for (int code : req.reference->codes)
                    if (code < 0 || code >= model.codec.codebook_size)
                        throw std::invalid_argument("invalid reference code");
                p.reference = req.reference;
                p.reference_text = tokenizer.encode(prefix + req.reference->text);
                has_reference = true;
            }
            p.target = tokenizer.encode(
                prefix +
                (use_instruction ? model.instruction_bos + req.instruction + model.instruction_eos
                                 : "") +
                req.text);
            for (const auto *ids : {&p.reference_text, &p.target})
                for (int id : *ids)
                    if (id < 0 || id >= model.text_vocab)
                        throw std::invalid_argument(
                            "tokenizer ID exceeds the text embedding vocabulary");
            if (p.length(model.codebooks) + opt.max_frames > context)
                throw std::invalid_argument("prompt plus max_frames exceeds context");
            maxprompt = std::max(maxprompt, p.length(model.codebooks));
            structured.push_back(std::move(p));
        }
    result.batch = batch;
    result.branches = branches;
    result.lanes = lanes;
    result.max_prompt = maxprompt;
    layout(result);
    for (int length : result.lengths)
        result.capacities.push_back(length + opt.max_frames);
    return result;
}
PreparedBatch pack_prefill(const std::vector<const PreparedBatch *> &inputs) {
    if (inputs.empty())
        throw std::invalid_argument("empty prefill batch");
    PreparedBatch result;
    for (const auto *input : inputs) {
        result.batch += input->batch;
        result.lanes += input->lanes;
        result.has_reference |= input->has_reference;
        result.max_prompt = std::max(result.max_prompt, input->max_prompt);
        result.prompts.insert(result.prompts.end(), input->prompts.begin(), input->prompts.end());
        result.capacities.insert(result.capacities.end(), input->capacities.begin(),
                                 input->capacities.end());
    }
    layout(result);
    return result;
}
namespace {
void layout(PreparedBatch &result) {
    result.offsets.push_back(0);
    for (int lane = 0; lane < result.lanes; ++lane) {
        const auto &p = result.prompts[lane];
        int length =
            int(p.reference_text.size() + p.target.size() +
                (p.reference ? p.reference->codes.size() / p.reference->codebooks + 1 : 0));
        result.lengths.push_back(length);
        int destination = result.prompt_rows;
        auto append = [&](const std::vector<int> &ids) {
            if (ids.empty())
                return;
            result.segments.push_back({int(result.tokens.size()), int(ids.size()), destination});
            result.tokens.insert(result.tokens.end(), ids.begin(), ids.end());
            result.max_text = std::max(result.max_text, int(ids.size()));
        };
        if (p.reference) {
            append(p.reference_text);
            destination +=
                p.reference_text.size() + p.reference->codes.size() / p.reference->codebooks + 1;
        }
        append(p.target);
        result.prompt_rows += length;
        result.offsets.push_back(result.prompt_rows);
    }
    result.text_rows = int(result.tokens.size());
    result.text_sequences = int(result.segments.size());
    result.storage_rows = (result.prompt_rows + 127) / 128 * 128;
    // Packed metadata: row -> sequence, row -> position, offsets, lengths.
    auto metadata = [](const std::vector<int> &lengths) {
        int rows = 0;
        for (int n : lengths)
            rows += n;
        int sequences = int(lengths.size()), row = 0;
        std::vector<int> data(2 * rows + 2 * sequences + 1);
        for (int lane = 0; lane < sequences; ++lane) {
            data[2 * rows + lane] = row;
            data[2 * rows + sequences + 1 + lane] = lengths[lane];
            for (int j = 0; j < lengths[lane]; ++j, ++row) {
                data[row] = lane;
                data[rows + row] = j;
            }
        }
        data[2 * rows + sequences] = row;
        return data;
    };
    std::vector<int> segment_lengths;
    for (auto segment : result.segments)
        segment_lengths.push_back(segment.size);
    result.text_metadata = metadata(segment_lengths);
    if (result.has_reference)
        result.backbone_metadata = metadata(result.lengths);
}
} // namespace
} // namespace tts
