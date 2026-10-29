#include "model.h"
#include "json.h"
#include <set>
namespace tts {
namespace {
SafeTensors load_checkpoint(const std::string &dir) {
    SafeTensors files;
    auto index = read_json(dir + "/model.safetensors.index.json");
    std::set<std::string> shards;
    for (const auto &[n, s] : index.at("weight_map").items())
        shards.insert(s.get<std::string>());
    for (const auto &s : shards) {
        if (s.empty() || s.find_first_of("/\\:") != std::string::npos ||
            s.find("..") != std::string::npos)
            throw std::runtime_error("invalid shard path");
        files.open_file(dir + "/" + s);
    }
    files.open_file(dir + "/audio_tokenizer/model.safetensors", "codec.");
    return files;
}
} // namespace

Model::Model(Backend &gpu, const std::string &dir, ModelConfig cfg)
    : config(std::move(cfg)), weights(gpu, load_checkpoint(dir)),
      tokenizer(dir + "/tokenizer.json"), text(weights, "text_encoder", config.text),
      backbone(weights, "backbone_model", config.backbone),
      depth(weights, "depth_decoder.model", config.depth), codec(weights, config.codec),
      encoder(weights, config.encoder_positions) {
    text_embedding = weights.get("text_encoder.embed_tokens.weight");
    eoi = weights.get("text_encoder.embed_tokens.eoi_embedding");
    text_proj = weights.get("text_encoder_proj.weight");
    audio_embedding = weights.get("depth_decoder.model.embed_tokens.weight");
    depth_proj = weights.get("depth_decoder.model.inputs_embeds_projector.weight");
    lm_head = weights.get("lm_head.weight");
    // Frozen embedding -> projection is input-independent: evaluate once at load.
    depth_embedding_storage = Tensor({audio_embedding.rows, depth.config.hidden}, depth.config.type, &gpu.memory);
    depth_embedding = depth_embedding_storage.view();
    gpu.linear(audio_embedding, depth_proj, depth_embedding);
    const auto &e = weights.entry("depth_decoder.codebooks_head.weight");
    int head_count = config.codebooks - 1, hidden = depth.config.hidden;
    int vocabulary = audio_embedding.rows / config.codebooks;
    if (audio_embedding.rows % config.codebooks ||
        e.shape != std::vector<int>({head_count, hidden, vocabulary}) || e.type != Type::BF16)
        throw std::runtime_error("depth head shape");
    const uint16_t *src = static_cast<const uint16_t *>(e.data);
    std::vector<uint16_t> transposed(vocabulary * hidden);
    for (int i = 0; i < head_count; i++) {
        for (int n = 0; n < vocabulary; n++)
            for (int k = 0; k < hidden; k++)
                transposed[n * hidden + k] = src[(size_t(i) * hidden + k) * vocabulary + n];
        depth_heads.push_back(weights.custom("depth_head." + std::to_string(i), transposed.data(),
                                             vocabulary, hidden, Type::BF16));
    }
    weights.seal();
    gpu.sync();
}
} // namespace tts
