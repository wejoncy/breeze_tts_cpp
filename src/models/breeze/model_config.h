#pragma once
#include "models/mimi/config.h"
#include "io/json.h"
#include <string>

namespace tts::breeze {
inline constexpr int max_guidance_branches = 3; // Unconditional, reference and instruction.
// Parsed and validated before device/weight initialization. This describes the
// supported Breeze checkpoint, not a promise of arbitrary architecture support.
struct ModelConfig {
    TransformerConfig text, backbone, depth;
    CodecConfig codec;
    EncoderConfig encoder;
    int codebooks, sample_rate, samples_per_frame;
    int max_reference_frames, audio_vocab, backbone_eos, codebook_eos, text_eoi, text_vocab;
    std::string instruction_bos, instruction_eos;
};
ModelConfig parse_model_config(const Json &config, const Json &audio_config, int batch, int context,
                              Quantization quantization = Quantization::None);
} // namespace tts::breeze
