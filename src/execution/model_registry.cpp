#include "execution/model_runtime.h"
#include "inputs/input_registry.h"
#include "execution/runner_resources.h"
#include "models/breeze/model.h"
#include "models/gpt_sovits/model.h"
#include "text/frontend_assets.h"
#include "io/model_loader.h"
#include "runtime/device.h"
#include <algorithm>

namespace tts {
namespace {
ModelComponents breeze_model(DeviceContext &memory, const ModelLoader &loader, const ModelOptions &options) {
    ModelComponents parts;
    parts.parameters = std::make_unique<Weights>(Device::cuda(memory.device), options.quantization,
        loader.open({{"", ""}, {"audio_tokenizer", "codec."}},
                      {"tokenizer.json", "audio_tokenizer/config.json"}));
    auto model = std::make_unique<breeze::Model>(*parts.parameters,
        breeze::parse_model_config(loader.config(), read_json(loader.path("audio_tokenizer/config.json")),
                                   options.max_batch, options.context, options.quantization));
    auto &model_instance = *model;
    parts.parameters->load(); parts.parameters->seal();
    parts.info = {"breeze", {model_instance.config.sample_rate, model_instance.config.samples_per_frame,
                   model_instance.config.codebooks, model_instance.config.codec.codebook_size},
                  std::min(options.context - 1, model_instance.codec.config.transformer.max_positions),
                  parts.parameters->size(), size_t(model_instance.config.max_reference_frames) * model_instance.config.samples_per_frame,
                  parts.parameters->checkpoint_files()};
    parts.quantum_ms = 1000. * parts.info.audio.samples_per_frame / parts.info.audio.sample_rate;
    parts.planner = load_inputs(loader, options.max_batch, options.context,
                                model_instance.backbone.config.type);
    parts.resources = std::make_unique<RunnerResources>(memory,
        RunnerResources::Cache{model_instance.backbone.config.layers,
            model_instance.backbone.config.kvheads * model_instance.backbone.config.dim,
            breeze::max_guidance_branches, model_instance.backbone.config.type},
        options.max_batch, options.context, options.kv_page_tokens);
    const auto &configuration = model_instance.config;
    StatePool::Layout state;
    state.tensors = {
        {"codes", {1, configuration.codebooks}, Type::I32},
        {"done", {1, 1}, Type::I32},
        {"positions", {breeze::max_guidance_branches, 1}, Type::I32},
        {"rng", {1, 1}, Type::I64},
        {"seen", {configuration.codec.codebook_size, 1}, Type::I32},
        {"table", {breeze::max_guidance_branches, parts.resources->kv.block_stride()}, Type::I32}};
    state.histories.emplace_back("audio", model_instance.codec.history_spec());
    parts.resources->states.prepare(state, options.max_batch);
    parts.network = std::move(model);
    return parts;
}
ModelComponents conditioned_model(DeviceContext &memory, const ModelLoader &loader, const ModelOptions &options) {
    auto checkpoint = loader.open({{"", ""}},
        {"tokenizer.json", "g2pw_tokenizer.json", "g2pw.json", "jieba.dict.utf8", "jieba.hmm.utf8", "pinyin.json",
         "polyphone.json", "linguistics.json", "english.json", "english.safetensors"});
    text::FrontendAssets assets(loader);
    checkpoint.files.insert(checkpoint.files.end(), assets.files().begin(), assets.files().end());
    std::sort(checkpoint.files.begin(), checkpoint.files.end());
    checkpoint.files.erase(std::unique(checkpoint.files.begin(), checkpoint.files.end()), checkpoint.files.end());
    ModelComponents parts;
    parts.parameters = std::make_unique<Weights>(Device::cuda(memory.device), options.quantization, std::move(checkpoint));
    auto model = std::make_unique<gpt_sovits::Model>(*parts.parameters, loader.config(),
        read_json(loader.path("g2pw.json")), options.context);
    auto &model_instance = *model;
    parts.parameters->load(); parts.parameters->seal();
    parts.info = {"gpt_sovits_v5", {model_instance.config.at("sample_rate"), model_instance.config.at("samples_per_frame"), 0, 0},
        model_instance.acoustic.feature_frames(options.context - 1), parts.parameters->size(),
        size_t(model_instance.config.at("reference").at("max_seconds").get<float>() *
               model_instance.config.at("sample_rate").get<int>()), parts.parameters->checkpoint_files()};
    parts.defaults.top_k = 15;
    parts.defaults.temperature = 1;
    parts.defaults.repetition_penalty = 1.35f;
    parts.defaults.max_frames = std::min(parts.info.max_output_frames, 2500);
    parts.defaults.sample_steps = model_instance.flow.sampling().steps;
    parts.defaults.cfg_rate = model_instance.flow.sampling().guidance;
    parts.quantum_ms = 1000. / model_instance.config.at("semantic_rate").get<int>();
    parts.planner = load_inputs(loader, options.max_batch, options.context, model_instance.semantic.type());
    parts.resources = std::make_unique<RunnerResources>(memory,
        RunnerResources::Cache{model_instance.semantic.layers(), model_instance.semantic.hidden(),
                               1, model_instance.semantic.type()},
        options.max_batch, options.context, options.kv_page_tokens);
    const Type state_type = model_instance.semantic.type();
    const int frames = model_instance.acoustic.feature_capacity(options.context);
    StatePool::Layout state;
    state.tensors = {
        {"table", {parts.resources->kv.block_stride(), 1}, Type::I32},
        {"codes", {options.context, 1}, Type::I32},
        {"positions", {1, 1}, Type::I32},
        {"phones", {options.context, 1}, Type::I32},
        {"seen", {model_instance.config.at("semantic").at("EOS"), 1}, Type::I32},
        {"rng", {1, 1}, Type::I64}, {"done", {1, 1}, Type::I32},
        {"waveform_condition", {1, model_instance.acoustic.waveform_condition_channels()}, state_type},
        {"style", {1, model_instance.config.at("acoustic").at("gin_channels")}, state_type},
        {"reference_mel", {model_instance.reference_frames, model_instance.mel_channels}, state_type},
        {"reference_features", {model_instance.reference_frames, model_instance.feature_channels}, state_type},
        {"features", {frames, model_instance.feature_channels}, state_type},
        {"text_condition", {options.context, model_instance.acoustic.text_condition_channels()}, state_type},
        {"reference_bert", {options.context, model_instance.semantic.hidden()}, state_type},
        {"history", {frames, model_instance.mel_channels}, state_type},
        {"offline_mel", {frames, model_instance.mel_channels}, state_type},
        {"feature_history", {frames, model_instance.feature_channels}, state_type},
        {"text_ready", {1, 1}, Type::I32, true},
        {"feature_tokens", {1, 1}, Type::I32, true}};
    parts.resources->states.prepare(state, options.max_batch);
    parts.network = std::move(model);
    return parts;
}
struct RegisteredModel {
    const char *type;
    ModelFactory build;
};
constexpr RegisteredModel models[] = {{"breeze", breeze_model},
                                      {"gpt_sovits_v5", conditioned_model}};
} // namespace
std::unique_ptr<ModelRuntime> load_model(const std::string &directory,
                                         const ModelOptions &options) {
    if (options.device < 0 || options.max_batch < 1 || options.context < 1 ||
        options.kv_page_tokens < 1)
        throw std::invalid_argument("invalid model runtime limits");
    const ModelLoader loader(directory);
    const auto type = loader.config().at("model_type").get<std::string>();
    for (const auto &factory : models)
        if (type == factory.type)
            return std::make_unique<ModelRuntime>(loader, options, factory.build);
    throw std::invalid_argument("unsupported model_type: " + type);
}
} // namespace tts
