#pragma once
#include "core/buffer.h"
#include "models/hubert/model.h"
#include "layers/audio/style_encoder.h"
#include "layers/spectrogram.h"
#include "core/forward_context.h"

namespace tts {
class ReferenceEncoder {
    Hubert hubert_;
    Convolution projection_;
    TensorView codebook_, norms_;
    StyleEncoder style_;
    Spectrogram spectrum_, mel_;
    int max_wave_samples_, max_hubert_samples_, style_bins_;
    Type type_;

  public:
    struct Output {
        TensorView codes, style, mel;
    };
    ReferenceEncoder(Weights &, const std::string &speech_prefix, const std::string &acoustic_prefix,
                     const Json &speech_config, const Json &acoustic_config,
                     const Json &reference_config, int mel_channels);
    Output forward(TensorView waveform, TensorView speech_waveform, ForwardContext &) const;

};
} // namespace tts
