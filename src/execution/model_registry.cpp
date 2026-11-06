#include "execution/model_registry.h"
#include "io/model_loader.h"
#include "support/init_log.h"

namespace tts {
const ModelRegistration *&ModelRegistration::head() {
    static const ModelRegistration *first = nullptr;
    return first;
}
ModelRegistration::ModelRegistration(const char *type, ModelFactory build) noexcept
    : type_(type), build_(build), next_(head()) {
    head() = this;
}
ModelFactory ModelRegistration::resolve(const std::string &type) {
    ModelFactory found = nullptr;
    for (auto *entry = head(); entry; entry = entry->next_) {
        if (!entry->type_ || !*entry->type_ || !entry->build_)
            throw std::logic_error("invalid model registration");
        if (type != entry->type_)
            continue;
        if (found)
            throw std::logic_error("duplicate model registration: " + type);
        found = entry->build_;
    }
    if (!found)
        throw std::invalid_argument("unsupported or disabled model_type: " + type);
    return found;
}
std::unique_ptr<ModelRuntime> load_model(const std::string &directory,
                                         const ModelOptions &options) {
    if (options.device < 0 || options.max_batch < 1 || options.context < 1 ||
        options.kv_page_tokens < 1)
        throw std::invalid_argument("invalid model runtime limits");
    log::InitPhase phase("model configuration");
    const ModelLoader loader(directory);
    auto factory = ModelRegistration::resolve(loader.config().at("model_type").get<std::string>());
    phase.next("model runtime");
    return std::make_unique<ModelRuntime>(loader, options, factory);
}
} // namespace tts
