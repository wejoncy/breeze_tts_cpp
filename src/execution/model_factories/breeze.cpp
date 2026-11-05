#include "execution/model_registry.h"
#include "execution/runner_resources.h"
#include "io/model_loader.h"
#include "runtime/device.h"
#include "inputs/token_inputs.h"
#include "models/breeze/model.h"
#include <algorithm>

namespace tts {
namespace {
ModelComponents breeze_model(DeviceContext &memory, const ModelLoader &loader,
                             const ModelOptions &options) {
    ModelComponents parts;
    parts.parameters =
        std::make_unique<Weights>(Device::cuda(memory.device), options.quantization,
                                  loader.open({{"", ""}, {"audio_tokenizer", "codec."}},
                                              {"tokenizer.json", "audio_tokenizer/config.json"}));
    auto model = std::make_unique<breeze::Model>(
        *parts.parameters,
        breeze::parse_model_config(loader.config(),
                                   read_json(loader.path("audio_tokenizer/config.json")),
                                   options.max_batch, options.context, options.quantization));
    auto &model_instance = *model;
    parts.parameters->load();
    parts.parameters->seal();
    parts.info = {
        "breeze",
        {model_instance.config.sample_rate, model_instance.config.samples_per_frame,
         model_instance.config.codebooks, model_instance.config.codec.codebook_size},
        std::min(options.context - 1, model_instance.codec.config.transformer.max_positions),
        parts.parameters->size(),
        size_t(model_instance.config.max_reference_frames) *
            model_instance.config.samples_per_frame,
        parts.parameters->checkpoint_files()};
    parts.quantum_ms = 1000. * parts.info.audio.samples_per_frame / parts.info.audio.sample_rate;
    parts.planner = std::make_unique<TokenInputs>(loader.path(""), model_instance.config,
                                                  options.max_batch, options.context);
    parts.resources = std::make_unique<RunnerResources>(
        memory,
        RunnerResources::Cache{model_instance.backbone.config.layers,
                               model_instance.backbone.config.kvheads *
                                   model_instance.backbone.config.dim,
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
const ModelRegistration registration("breeze", breeze_model);
} // namespace
} // namespace tts
