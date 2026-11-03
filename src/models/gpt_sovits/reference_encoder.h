#pragma once
#include "models/gpt_sovits/speech_encoder.h"
#include "models/gpt_sovits/style_encoder.h"
#include "layers/spectrogram.h"

namespace tts::gpt_sovits {
class ReferenceEncoder {
    Hubert hubert_;
    Convolution projection_;
    TensorView codebook_, norms_;
    StyleEncoder style_;
    Spectrogram spectrum_, mel_;
    int max_wave_samples_, max_hubert_samples_, style_bins_, style_channels_;
    Type type_;

  public:
    struct Workspace {
        Hubert::Workspace hubert;
        StyleEncoder::Workspace style_encoder;
        Spectrogram::Workspace spectrum, mel_spectrum;
        Buffer projected, convolution, scores, codes, magnitude, style, mel;
        explicit Workspace(const ReferenceEncoder &, BufferAllocator *);
    };
    struct Output {
        TensorView codes, style, mel;
    };
    ReferenceEncoder(Weights &, const Json &);
    void prepare()const { hubert_.prepare(max_hubert_samples_); }
    Output forward(Workspace &, TensorView wave, TensorView hubert_wave) const;
};
} // namespace tts::gpt_sovits
