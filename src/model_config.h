#pragma once
#include "audio_config.h"
#include <string>

namespace tts {
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
ModelConfig load_model_config(const std::string &dir, int batch, int context,
                              Quantization quantization = Quantization::None);
} // namespace tts
