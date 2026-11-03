#include "models/gpt_sovits/model.h"
#include "io/json.h"
#include "models/gpt_sovits/model_runner.h"
#include "models/gpt_sovits/request_planner.h"
#include "models/gpt_sovits/frontend_assets.h"
#include <algorithm>
#include <cmath>

namespace tts::gpt_sovits {
namespace {
ModelLoader::Checkpoint checkpoint(const ModelLoader &loader) {
    auto checkpoint = loader.open({{"", ""}},
        {"tokenizer.json", "g2pw_tokenizer.json", "g2pw.json", "jieba.bin", "pinyin.bin",
         "polyphone.bin", "linguistics.json", "english.json", "english.safetensors"});
    FrontendAssets assets(loader);
    checkpoint.files.insert(checkpoint.files.end(), assets.files().begin(), assets.files().end());
    std::sort(checkpoint.files.begin(), checkpoint.files.end());
    checkpoint.files.erase(std::unique(checkpoint.files.begin(), checkpoint.files.end()),
                           checkpoint.files.end());
    return checkpoint;
}
} // namespace
Model::Model(Backend &gpu, const ModelLoader &loader, int context)
    : config(loader.config()),
      weights(gpu, checkpoint(loader)),
      bert(weights, config.at("bert"), "bert.bert",
           config.at("bert").at("num_hidden_layers").get<int>() - 2, gpu.has_quantization()),
      pronunciation(weights, read_json(loader.path("g2pw.json"))),
      semantic(weights, config.at("semantic"), context), reference(weights, config),
      acoustic(weights, config.at("acoustic")),
      flow(weights, config.at("flow"), config.at("synthesis").at("window_frames")),
      vocoder(weights, config.at("vocoder")),
      window_frames(config.at("synthesis").at("window_frames")),
      reference_frames(config.at("synthesis").at("reference_frames")),
      rolling_frames(config.at("synthesis").at("rolling_frames")),
      target_frames(config.at("synthesis").at("target_frames")),
      mel_channels(config.at("flow").at("mel_channels")),
      feature_channels(config.at("flow").at("text_channels")),
      mel_minimum(config.at("reference").at("mel_min")),
      mel_maximum(config.at("reference").at("mel_max")) {
    if (config.at("model_type") != "gpt_sovits_v5" || window_frames < 2 || reference_frames < 1 ||
        reference_frames >= window_frames || rolling_frames < 0 ||
        rolling_frames > reference_frames || target_frames < 1 || target_frames > window_frames ||
        !(mel_maximum > mel_minimum) || !std::isfinite(mel_minimum) ||
        !std::isfinite(mel_maximum) || vocoder.samples(1) != config.at("samples_per_frame"))
        throw std::invalid_argument("inconsistent GPT-SoVITS synthesis configuration");
}
void Model::prepare_flow(Flow::Workspace &w, const DiT::Condition &condition,
                           const ForwardBatch &input, const int *prefix, const uint64_t *seeds,
                           const uint64_t *windows, FlowSampling parameters) const {
    flow.prepare(w, condition, input, parameters);
    flow.initialize(w, input, prefix, seeds, windows);
}
void Model::flow_step(Flow::Workspace &w, const ForwardBatch &input,
                      const int *prefix_lengths, int prefix, int step, FlowSampling parameters,
                      const ops::EulerOutput &output) const {
    int target = input.time() - prefix;
    if (prefix < 0 || target < 1 || target > target_frames)
        throw std::invalid_argument("CFM target exceeds initialized window");
    flow.step(w, input, prefix_lengths, step, parameters, output);
}

namespace {
class Runtime final : public ModelRuntime {
    Backend gpu_;
    Model model_;
    std::unique_ptr<Planner> planner_;
    std::unique_ptr<Runner> runner_;
    const ModelInfo info_;

  public:
    Runtime(const ModelLoader &loader, const ModelOptions &options)
        : gpu_(options.device, options.quantization, options.memory_limit_bytes),
          model_(gpu_, loader, options.context),
          info_{"gpt_sovits_v5",
                {model_.config.at("sample_rate"), model_.config.at("samples_per_frame"), 0, 0},
                model_.acoustic.feature_frames(options.context - 1),
                model_.weights.size(),
                size_t(model_.config.at("reference").at("max_seconds").get<float>() *
                       model_.config.at("sample_rate").get<int>()),
                model_.weights.checkpoint_files()} {
        model_.weights.load();
        model_.weights.seal();
        planner_ = std::make_unique<Planner>(loader.path(""), model_.config, options.context,
                                            model_.semantic.type());
        runner_ = std::make_unique<Runner>(model_, gpu_, options.max_batch, options.context,
                                          options.kv_page_tokens, options.runtime_memory_bytes);
    }
    ~Runtime() override {
        cudaSetDevice(gpu_.device);
        cudaStreamSynchronize(gpu_.stream);
    }
    const ModelInfo &info() const override {
        return info_;
    }
    GenerationOptions default_options() const override {
        GenerationOptions options;
        options.top_k = 15;
        options.temperature = 1;
        options.repetition_penalty = 1.35f;
        options.max_frames = std::min(info_.max_output_frames, 2500);
        options.sample_steps = model_.flow.sampling().steps;
        options.cfg_rate = model_.flow.sampling().guidance;
        return options;
    }
    RequestPlanner &planner() override {
        return *planner_;
    }
    ModelRunner &runner() override {
        return *runner_;
    }
    double scheduling_quantum_ms() const override {
        return 1000. / model_.config.at("semantic_rate").get<int>();
    }
    MemoryStats memory_stats() const override {
        gpu_.memory.sync();
        return gpu_.memory.stats();
    }
};
} // namespace
std::unique_ptr<ModelRuntime> load_model(const ModelLoader &loader,
                                         const ModelOptions &options) {
    return std::make_unique<Runtime>(loader, options);
}
} // namespace tts::gpt_sovits
