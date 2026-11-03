#pragma once
#include "io/json.h"
#include "io/safetensors.h"
#include <initializer_list>

namespace tts {
// Checkpoint discovery is independent of models, GPU storage and execution.
class ModelLoader {
    const std::string directory_;
    const Json config_;

  public:
    struct Source {
        std::string subdirectory, prefix;
    };
    struct Checkpoint {
        SafeTensors tensors;
        std::vector<std::string> files;
    };
    explicit ModelLoader(const std::string &directory);
    const Json &config() const { return config_; }
    std::string path(const std::string &relative) const;
    Checkpoint open(std::initializer_list<Source> sources,
                    std::initializer_list<const char *> assets) const;
};
} // namespace tts
