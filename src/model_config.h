#pragma once
#include "transformer_config.h"
#include <string>

namespace tts {
// Parsed and validated before device/weight initialization. This describes the
// supported Breeze checkpoint, not a promise of arbitrary architecture support.
struct ModelConfig {
    TransformerConfig text, backbone, depth, codec;
    int codebooks, sample_rate, samples_per_frame;
    int encoder_positions, max_reference_frames;
};
ModelConfig load_model_config(const std::string &dir, int batch, int context,
                              Quantization quantization = Quantization::None);
} // namespace tts
