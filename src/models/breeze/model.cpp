#include "ops/basic.h"
#include "models/breeze/model.h"
#include "io/json.h"
#include "layers/embedding.h"
#include "layers/linear.h"
#include <cmath>
namespace tts::breeze {
Model::Model(Weights &weights, ModelConfig configuration)
    : config(std::move(configuration)),
      text(weights, "text_encoder", config.text),
      backbone(weights, "backbone_model", config.backbone),
      depth(weights, "depth_decoder.model", config.depth),
      codec(weights, "codec.decoder", config.codec),
      encoder(weights, "codec.encoder", config.encoder),
      sampler({config.codebooks, config.codec.codebook_size, config.backbone_eos}) {
    text_embedding = weights.get("text_encoder.embed_tokens.weight");
    eoi = weights.get("text_encoder.embed_tokens.eoi_embedding");
    text_proj = weights.get("text_encoder_proj.weight");
    audio_embedding = weights.get("depth_decoder.model.embed_tokens.weight");
    depth_proj = weights.get("depth_decoder.model.inputs_embeds_projector.weight");
    lm_head = weights.get("lm_head.weight");
    auto shape = [](TensorView tensor, int rows, int cols) {
        if (tensor.type != Type::BF16 || tensor.rows != rows || tensor.cols != cols)
            throw std::invalid_argument("model projection/embedding weight/configuration mismatch");
    };
    shape(text_embedding, config.text_vocab, config.text.hidden);
    shape(text_proj, config.backbone.hidden, config.text.hidden);
    shape(audio_embedding, config.codebooks * config.audio_vocab, config.backbone.hidden);
    shape(depth_proj, config.depth.hidden, config.backbone.hidden);
    shape(lm_head, config.backbone_eos + 1, config.backbone.hidden);
    if (eoi.type != Type::BF16 || eoi.numel() != size_t(config.text.hidden))
        throw std::invalid_argument("text replacement embedding shape");
    // Frozen embedding -> projection is input-independent: evaluate once at load.
    depth_embedding_storage = Tensor({audio_embedding.rows, depth.config.hidden}, depth.config.type,
                                      weights.device());
    depth_embedding = depth_embedding_storage.view();
    const auto &checkpoint_head = weights.entry("depth_decoder.codebooks_head.weight");
    int head_count = config.codebooks - 1, hidden = depth.config.hidden;
    int vocabulary = audio_embedding.rows / config.codebooks;
    if (audio_embedding.rows % config.codebooks ||
        checkpoint_head.shape != std::vector<int>({head_count, hidden, vocabulary}) ||
        checkpoint_head.type != Type::BF16)
        throw std::runtime_error("depth head shape");
    const uint16_t *source = static_cast<const uint16_t *>(checkpoint_head.data);
    std::vector<uint16_t> transposed(vocabulary * hidden);
    for (int head_index = 0; head_index < head_count; head_index++) {
        for (int token_index = 0; token_index < vocabulary; token_index++)
            for (int hidden_index = 0; hidden_index < hidden; hidden_index++)
                transposed[token_index * hidden + hidden_index] =
                    source[(size_t(head_index) * hidden + hidden_index) * vocabulary + token_index];
        depth_heads.push_back(weights.custom("depth_head." + std::to_string(head_index),
                                             transposed.data(), vocabulary, hidden, Type::BF16));
    }
    residual_codebooks = std::make_unique<ResidualCodebooks>(depth, depth_embedding, depth_proj,
        depth_heads, SamplingLayout{config.codebooks, config.codec.codebook_size, config.backbone_eos});
}
void Model::prepare_weights() const {
    Linear(depth_proj).forward(audio_embedding, depth_embedding);
}
void Model::forward(std::string_view program, TensorInputs inputs, TensorOutputs outputs,
                     ForwardContext &context) const {
    if (program == "tokens") {
        auto hidden_states = context.tensor({context.max_batch, backbone.config.hidden},
                                             backbone.config.type).slice(0, context.batch.batch());
        if (!context.batch.is_decode()) {
            Tensor::Scope prompt_scope;
            const int prompt_rows = context.mode == ForwardContext::Prepare ? 1 : context.batch.rows();
            auto projected = context.tensor({prompt_rows, backbone.config.hidden},
                                            backbone.config.type);
            {
                Tensor::Scope text_scope;
                auto text_context = context;
                const auto text_layout = context.sequences["text"];
                text_context.batch = ForwardBatch(text_layout);
                text_context.batch.fixed_position = 0;
                text_context.max_batch = text_layout.sequences;
                text_context.max_time = text_layout.max_length;
                text_context.max_rows = text_layout.rows;
                auto encoded = Embedding(text_embedding, std::sqrt(float(text.config.hidden)),
                                          eoi, config.text_eoi).forward(inputs["tokens"], text_context);
                encoded = text.forward(encoded, text_context);
                Linear(text_proj).forward_segments(encoded, inputs["segments"], projected, context);
                Embedding(audio_embedding).forward_codebooks(context.regions, projected,
                                                               config.codebooks, context);
            }
            auto backbone_context = context;
            backbone_context.batch.output = hidden_states;
            hidden_states = backbone.forward(projected, backbone_context);
        } else {
            auto embedded = Embedding(audio_embedding).forward_codebooks(
                inputs["previous_codes"], inputs["owner_rows"], config.codebooks, context);
            auto backbone_context = context;
            backbone_context.batch.output = hidden_states;
            hidden_states = backbone.forward(embedded, backbone_context);
        }
        auto head_context = context;
        head_context.max_rows = context.max_batch;
        // Keep the FP32 accumulator through CFG and sampling, as in the official runtime.
        auto logits = head_context.tensor({head_context.max_rows, lm_head.rows}, Type::F32)
                          .slice(0, hidden_states.rows);
        Linear(lm_head).forward(hidden_states, logits, head_context);
        auto done = sampler.forward(logits, inputs["sampling"], inputs["done"], head_context);
        outputs["logits"] = logits;
        outputs["hidden_states"] = hidden_states;
        outputs["done"] = residual_codebooks->forward(hidden_states, inputs["code_rows"],
            inputs["owner_rows"], inputs["sampling"], done, head_context);
    } else if (program == "waveform") {
        codec.forward(inputs, outputs, context);
    } else if (program == "reference_codes") {
        outputs["codes"] = encoder.forward(inputs["waveform"], context);
    } else {
        throw std::invalid_argument("unknown model forward program");
    }
}

} // namespace tts::breeze
