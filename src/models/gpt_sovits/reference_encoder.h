#pragma once
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
    int max_wave_samples_, max_hubert_samples_, style_bins_, style_channels_;
    Type type_;

    struct Workspace {
        Hubert::Workspace hubert;
        StyleEncoder::Workspace style_encoder;
        Spectrogram::Workspace spectrum, mel_spectrum;
        Buffer projected, convolution, scores, codes, magnitude, style, mel;
        explicit Workspace(const ReferenceEncoder &, BufferAllocator *);
    };

  public:
    struct Output {
        TensorView codes, style, mel;
    };
    ReferenceEncoder(Weights &, const std::string &speech_prefix, const std::string &acoustic_prefix,
                     const Json &speech_config, const Json &acoustic_config,
                     const Json &reference_config, int mel_channels);
    Output forward(TensorView waveform, TensorView speech_waveform, ForwardContext &) const;

  private:
    Output forward(Workspace &, TensorView wave, TensorView hubert_wave) const;
};
} // namespace tts
