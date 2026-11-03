#pragma once
#include "tts/request.h"

namespace tts {
struct ModelInfo {
    std::string model_type;
    AudioFormat audio;
    int max_output_frames;
    size_t weight_bytes, max_reference_samples;
    // Relative files actually used by the loaded model, for checkpoint identity.
    std::vector<std::string> checkpoint_files;
};
} // namespace tts
