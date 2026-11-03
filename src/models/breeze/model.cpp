#include "models/breeze/model.h"
#include "models/breeze/model_runner.h"
#include "models/breeze/generation_state.h"
#include "support/tensor_debug.h"
#include "models/breeze/request_planner.h"
#include "io/json.h"
namespace tts::breeze {
Model::Model(Backend &gpu, const ModelLoader &loader, ModelConfig cfg)
    : config(std::move(cfg)),
      weights(gpu, loader.open({{"", ""}, {"audio_tokenizer", "codec."}},
                                {"tokenizer.json", "audio_tokenizer/config.json"})),
      tokenizer(loader.path("tokenizer.json")), text(weights, "text_encoder", config.text),
      backbone(weights, "backbone_model", config.backbone),
      depth(weights, "depth_decoder.model", config.depth), codec(weights, config.codec),
      encoder(weights, config.encoder) {
    text_embedding = weights.get("text_encoder.embed_tokens.weight");
    eoi = weights.get("text_encoder.embed_tokens.eoi_embedding");
    text_proj = weights.get("text_encoder_proj.weight");
    audio_embedding = weights.get("depth_decoder.model.embed_tokens.weight");
    depth_proj = weights.get("depth_decoder.model.inputs_embeds_projector.weight");
    lm_head = weights.get("lm_head.weight");
    auto shape = [](TensorView v, int rows, int cols) {
        if (v.type != Type::BF16 || v.rows != rows || v.cols != cols)
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
    depth_embedding_storage = Tensor({audio_embedding.rows, depth.config.hidden}, depth.config.type, &gpu.memory);
    depth_embedding = depth_embedding_storage.view();
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
    weights.after_loading([input = audio_embedding, projection = depth_proj,
                           output = depth_embedding](Backend &backend) {
        backend.linear(input, projection, output);
    });
}

void Model::depth_frame(GenerationState &s, int batch, const DecodeBatch &mapped,
                              bool debug) const {
    auto &gpu = backend_for(s.last.device());
    int lanes = mapped.lanes;
    auto &g = gpu;
    auto h = s.last.view(lanes, backbone.config.hidden),
         l = s.logits.view(lanes, lm_head.rows);
    {
        Range r("tts.lm_head");
        g.linear(h, lm_head, l);
        tts::sample(
            g.stream, l,
            {config.codebooks, config.codec.codebook_size, config.backbone_eos},
            mapped.sampling, batch, 0, s.done.ptr<int>());
    }
    if (debug)
        dump_tensor(g, l, ".backbone_logits.bf16");
    Range range("tts.depth");
    auto hidden = s.depth_hidden.view(lanes * 2, depth.config.hidden),
         dl = s.depth_logits.view(lanes, depth_heads[0].rows);
    auto first = hidden.every_rows(0, 2, lanes);
    auto dh = hidden.every_rows(1, 2, lanes);
    g.linear(h, depth_proj, first);
    tts::audio_embedding(g.stream, depth_embedding, nullptr, dh, lanes, 1,
                         config.codebooks, 0, mapped.codes, mapped.row_request);
    ForwardBatch depth_input{lanes, 2};
    depth_input.fixed_position = 0;
    depth.forward(s.ds, hidden, depth_input);
    for (int cb = 1; cb < config.codebooks; cb++) {
        if (cb > 1) {
            hidden = s.depth_hidden.view(lanes, depth.config.hidden);
            dh = hidden;
            tts::audio_embedding(g.stream, depth_embedding, nullptr, hidden, lanes, 1,
                                 config.codebooks, cb - 1, mapped.codes, mapped.row_request);
            depth_input.sequence = SequenceLayout::uniform(lanes, 1);
            depth_input.fixed_position = cb;
            depth.forward(s.ds, hidden, depth_input);
        }
        g.linear(dh, depth_heads[cb - 1], dl);
        if (debug)
            dump_tensor(g, dl, ".depth_" + std::to_string(cb) + ".bf16");
        tts::sample(
            g.stream, dl,
            {config.codebooks, config.codec.codebook_size, config.backbone_eos},
            mapped.sampling, batch, cb, s.done.ptr<int>());
    }
}
void Model::next_frame(GenerationState &s, int batch, const DecodeBatch &mapped) const {
    auto &gpu = backend_for(s.last.device());
    int lanes = mapped.lanes;
    {
        Range r("tts.backbone");
        auto input = s.last.view(lanes, backbone.config.hidden);
        tts::audio_embedding(gpu.stream, audio_embedding, nullptr, input, lanes, 1,
                             config.codebooks, -1, mapped.codes, mapped.row_request);
        ForwardBatch batch_input{lanes, 1};
        batch_input.cache = s.bs.external_kv;
        batch_input.cache.layout.sequences = mapped.cache;
        backbone.forward(s.bs, input, batch_input);
        gpu.increment(nullptr, lanes, 1, mapped.cache);
    }
    depth_frame(s, batch, mapped);
}

namespace {
class Runtime final : public ModelRuntime {
    Backend gpu_;
    Model model_;
    std::unique_ptr<Planner> planner_;
    std::unique_ptr<Runner> runner_;
    const ModelInfo info_;

  public:
    Runtime(const ModelLoader &loader, const ModelOptions &options, ModelConfig config)
        : gpu_(options.device, options.quantization, options.memory_limit_bytes),
          model_(gpu_, loader, std::move(config)),
          info_{"breeze", {model_.config.sample_rate, model_.config.samples_per_frame,
                 model_.config.codebooks, model_.config.codec.codebook_size},
                std::min(options.context - 1, model_.codec.config.transformer.max_positions),
                model_.weights.size(),
                size_t(model_.config.max_reference_frames) * model_.config.samples_per_frame,
                model_.weights.checkpoint_files()} {
        model_.weights.load();
        model_.weights.seal();
        planner_ = std::make_unique<Planner>(model_.tokenizer, model_.config,
                                            options.max_batch, options.context);
        runner_ = std::make_unique<Runner>(model_, gpu_, options.max_batch, options.context,
                                          options.kv_page_tokens, options.runtime_memory_bytes);
    }
    ~Runtime() override {
        cudaSetDevice(gpu_.device);
        cudaStreamSynchronize(gpu_.stream);
    }
    const ModelInfo &info() const override { return info_; }
    GenerationOptions default_options() const override { return {}; }
    RequestPlanner &planner() override { return *planner_; }
    ModelRunner &runner() override { return *runner_; }
    AudioCodec *audio_codec() override { return runner_.get(); }
    double scheduling_quantum_ms() const override {
        return 1000. * info_.audio.samples_per_frame / info_.audio.sample_rate;
    }
    MemoryStats memory_stats() const override {
        gpu_.memory.sync();
        return gpu_.memory.stats();
    }
};
} // namespace
std::unique_ptr<ModelRuntime> load_model(const ModelLoader &loader, const ModelOptions &options) {
    return std::make_unique<Runtime>(loader, options,
        parse_model_config(loader.config(), read_json(loader.path("audio_tokenizer/config.json")),
                           options.max_batch, options.context, options.quantization));
}
} // namespace tts::breeze
