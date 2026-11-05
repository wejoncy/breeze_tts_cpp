#pragma once
#include "execution/model_runtime.h"

namespace tts {
// Registration objects have static storage and link during library initialization.
// Engine creation only reads the metadata; registration allocates no resources.
class ModelRegistration {
    const char *const type_;
    const ModelFactory build_;
    const ModelRegistration *const next_;
    static const ModelRegistration *&head();

  public:
    ModelRegistration(const char *type, ModelFactory build) noexcept;
    ModelRegistration(const ModelRegistration &) = delete;
    static ModelFactory resolve(const std::string &type);
};
} // namespace tts
