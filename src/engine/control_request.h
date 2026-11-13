#pragma once
#include "tts/request.h"
#include <variant>

namespace tts {
using ControlResult = std::variant<std::monostate, std::shared_ptr<const Reference>,
                                   std::vector<int>, std::vector<float>>;
struct ControlRequest {
    enum class Kind { Reference, Encode, Decode, Configure };
    Kind kind = Kind::Reference;
    std::vector<float> mono;
    std::vector<int> codes;
    std::string transcript, language;
    size_t request_limit = 0, output_bytes = 0;
    ControlResult result;
    std::exception_ptr error;
    bool complete = false;
};
} // namespace tts
