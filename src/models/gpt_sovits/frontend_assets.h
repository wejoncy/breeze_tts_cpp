#pragma once
#include "io/model_loader.h"
#include <filesystem>
#include <set>

namespace tts::gpt_sovits {
// Optional frontends are declared by the converted checkpoint. Missing assets
// disable that language without preventing other languages from loading.
class FrontendAssets {
    std::set<std::string> available_, files_;
  public:
    explicit FrontendAssets(const ModelLoader &loader) {
        const auto &config = loader.config();
        if (!config.contains("frontend")) return;
        const auto &languages = config.at("frontend").at("languages");
        if (!languages.is_object())
            throw std::invalid_argument("frontend languages must be an object");
        for (const auto &[language, assets] : languages.items()) {
            if (language.empty() || !assets.is_array() || assets.empty())
                throw std::invalid_argument("frontend language requires an asset list");
            bool complete = true;
            for (const auto &asset : assets) {
                auto name = asset.get<std::string>();
                if (name.empty()) throw std::invalid_argument("empty frontend asset path");
                if (std::filesystem::is_regular_file(std::filesystem::u8path(loader.path(name))))
                    files_.insert(std::filesystem::u8path(name).lexically_normal().generic_u8string());
                else complete = false;
            }
            if (complete) available_.insert(language);
        }
    }
    bool available(const std::string &language) const { return available_.count(language) != 0; }
    const std::set<std::string> &files() const { return files_; }
};
} // namespace tts::gpt_sovits
