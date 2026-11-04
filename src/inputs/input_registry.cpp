#include "inputs/input_registry.h"
#include "inputs/token_inputs.h"
#include "io/model_loader.h"

namespace tts {
std::unique_ptr<RequestPlanner> load_phoneme_inputs(const ModelLoader &, int batch, int context, Type);
std::unique_ptr<RequestPlanner> load_inputs(const ModelLoader &loader, int batch, int context, Type type) {
    const auto model_type = loader.config().at("model_type").get<std::string>();
    if (model_type == "breeze")
        return std::make_unique<TokenInputs>(loader.path(""),
            breeze::parse_model_config(loader.config(), read_json(loader.path("audio_tokenizer/config.json")),
                                       batch, context), batch, context);
    if (model_type == "gpt_sovits_v5") return load_phoneme_inputs(loader, batch, context, type);
    throw std::invalid_argument("no input processor registered for model_type: " + model_type);
}
} // namespace tts
