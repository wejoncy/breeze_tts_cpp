#pragma once
#include "tts/request.h"
#include "models/breeze/model_config.h"
#include "io/tokenizer.h"
namespace tts {
struct PreparedPrompt {
    std::vector<int> reference_text, target;
    std::shared_ptr<const Reference> reference;
    int length(int codebooks) const {
        return int(reference_text.size() + target.size() +
                   (reference ? reference->codes.size() / codebooks + 1 : 0));
    }
};
// CPU-owned inputs. Metadata and upload sources live until generation completes.
struct PreparedBatch {
    struct Segment {
        int start, size, destination;
    };
    int batch = 0, branches = 0, lanes = 0, max_prompt = 0, max_text = 0;
    int prompt_rows = 0, text_rows = 0, text_sequences = 0, storage_rows = 0;
    bool has_reference = false;
    std::vector<PreparedPrompt> prompts;
    std::vector<Segment> segments;
    std::vector<int> tokens, text_metadata, backbone_metadata, lengths, offsets, capacities;
};
PreparedBatch prepare_batch(const Tokenizer &, const ModelConfig &, int max_batch, int context,
                            const std::vector<Input> &, const GenerationOptions &);
// Pack independent, already validated inputs without tokenization or padding.
// Guidance/sampling stay with each request; this layout only addresses prefill rows.
PreparedBatch pack_prefill(const std::vector<const PreparedBatch *> &);
} // namespace tts
