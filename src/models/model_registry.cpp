#include "execution/model_runtime.h"
#include "models/breeze/model.h"
#include "models/gpt_sovits/model.h"
#include "io/model_loader.h"
#include <cmath>

namespace tts {
namespace {
struct ModelFactory {
    const char *type;
    std::unique_ptr<ModelRuntime> (*load)(const ModelLoader &, const ModelOptions &);
};
constexpr ModelFactory models[] = {{"breeze", breeze::load_model},
                                   {"gpt_sovits_v5", gpt_sovits::load_model}};
} // namespace
std::unique_ptr<ModelRuntime> load_model(const std::string &directory,
                                         const ModelOptions &options) {
    if (options.device < 0 || options.max_batch < 1 || options.context < 1 ||
        options.kv_page_tokens < 1)
        throw std::invalid_argument("invalid model runtime limits");
    const ModelLoader loader(directory);
    const auto type = loader.config().at("model_type").get<std::string>();
    for (const auto &factory : models) {
        if (type != factory.type)
            continue;
        auto runtime = factory.load(loader, options);
        const auto &info = runtime->info();
        const double quantum = runtime->scheduling_quantum_ms();
        if (info.model_type != type || info.audio.sample_rate < 1 ||
            info.audio.samples_per_frame < 1 || info.max_output_frames < 1 ||
            !std::isfinite(quantum) || quantum <= 0)
            throw std::invalid_argument("invalid runtime metadata for model_type: " + type);
        return runtime;
    }
    throw std::invalid_argument("unsupported model_type: " + type);
}
} // namespace tts
