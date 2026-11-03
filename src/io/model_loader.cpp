#include "io/model_loader.h"
#include <filesystem>
#include <set>

namespace tts {
ModelLoader::ModelLoader(const std::string &directory)
    : directory_(directory), config_(read_json(path("config.json"))) {}

std::string ModelLoader::path(const std::string &relative) const {
    const auto p = std::filesystem::u8path(relative);
    if (p.is_absolute() || p.has_root_name() || relative.find_first_of("\\:") != std::string::npos)
        throw std::invalid_argument("checkpoint path must be relative: " + relative);
    for (const auto &part : p)
        if (part == "..")
            throw std::invalid_argument("checkpoint path escapes model directory: " + relative);
    return (std::filesystem::u8path(directory_) / p).u8string();
}
ModelLoader::Checkpoint ModelLoader::open(std::initializer_list<Source> sources,
                                          std::initializer_list<const char *> assets) const {
    if (!sources.size()) throw std::invalid_argument("model has no weight sources");
    Checkpoint checkpoint;
    std::set<std::string> files;
    auto asset = [&](const std::string &relative) {
        if (!std::filesystem::is_regular_file(std::filesystem::u8path(path(relative))))
            throw std::runtime_error("missing checkpoint asset: " + relative);
        files.insert(std::filesystem::u8path(relative).lexically_normal().generic_u8string());
    };
    asset("config.json");
    for (const auto *name : assets) asset(name);
    for (const auto &source : sources)
        for (const auto &file : checkpoint.tensors.open_checkpoint(path(source.subdirectory),
                                                                  source.prefix))
            asset((std::filesystem::u8path(source.subdirectory) /
                   std::filesystem::u8path(file)).generic_u8string());
    checkpoint.files.assign(files.begin(), files.end());
    return checkpoint;
}
} // namespace tts
