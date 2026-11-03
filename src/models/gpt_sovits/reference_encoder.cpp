#include "models/gpt_sovits/reference_encoder.h"
#include "ops/vector_quantization.h"
#include <cmath>

namespace tts::gpt_sovits {
namespace {
SpectrogramConfig spectral_config(const Json &c, int channels) {
    int fft = c.at(channels ? "mel_fft" : "filter_length");
    int hop = c.at(channels ? "mel_hop" : "hop_length");
    int window = c.at(channels ? "mel_window" : "win_length");
    SpectrogramConfig result{fft, hop, window, (fft - hop) / 2,
                             c.at(channels ? "mel_sample_rate" : "sampling_rate")};
    result.mel_channels = channels;
    if (channels) {
        float low = c.at("mel_min"), high = c.at("mel_max");
        if (!(high > low))
            throw std::invalid_argument("invalid mel normalization range");
        result.log_scale = 2.f / (high - low);
        result.log_bias = -low * result.log_scale - 1;
    }
    return result;
}
int spectral_capacity(const Json &c, int channels) {
    auto geometry = spectral_config(c, channels);
    return geometry.frames(int(c.at("max_seconds").get<float>() * geometry.sample_rate));
}
ops::ConvGeometry projection_geometry(Weights &w, const Json &c) {
    const auto &shape = w.entry("acoustic.ssl_proj.weight").shape;
    int stride = c.at("semantic_frame_rate") == "25hz" ? 2 : 1;
    if (shape.size() != 3 || shape[2] != stride)
        throw std::invalid_argument("semantic feature projection stride mismatch");
    return {shape[1], shape[0], shape[2], stride};
}
TensorView codebook_norms(Weights &w) {
    const std::string name = "acoustic.quantizer.vq.layers.0._codebook.embed";
    auto table = w.get(name);
    auto values = w.host_values(name);
    auto round = [&](float x) {
        if (table.type == Type::F16)
            return float(__float2half_rn(x));
        if (table.type == Type::BF16)
            return float(__float2bfloat16_rn(x));
        return x;
    };
    std::vector<float> norms(table.rows, 0);
    for (int r = 0; r < table.rows; ++r) {
        double sum = 0;
        for (int d = 0; d < table.cols; ++d) {
            float v = values[size_t(r) * table.cols + d];
            sum += round(v * v);
        }
        norms[r] = round(float(sum));
    }
    return w.generated(name + ".norms", norms, 1, table.rows, table.type);
}
} // namespace
ReferenceEncoder::ReferenceEncoder(Weights &w, const Json &config)
    : hubert_(w, config.at("hubert")),
      projection_(w, "acoustic.ssl_proj", projection_geometry(w, config.at("acoustic"))),
      codebook_(w.get("acoustic.quantizer.vq.layers.0._codebook.embed")), norms_(codebook_norms(w)),
      style_(w, config.at("acoustic").at("style_heads"),
             config.at("acoustic").at("style_temporal_layers")),
      spectrum_(w.backend(), spectral_config(config.at("reference"), 0),
                spectral_capacity(config.at("reference"), 0)),
      mel_(w.backend(),
           spectral_config(config.at("reference"), config.at("flow").at("mel_channels")),
           spectral_capacity(config.at("reference"), config.at("flow").at("mel_channels"))),
      max_wave_samples_(int(config.at("reference").at("max_seconds").get<float>() *
                            config.at("reference").at("sampling_rate").get<int>())),
      max_hubert_samples_(int(config.at("reference").at("max_seconds").get<float>() *
                                  config.at("reference").at("hubert_sample_rate").get<int>() +
                              config.at("reference").at("semantic_silence_seconds").get<float>() *
                                  config.at("reference").at("sampling_rate").get<int>())),
      style_bins_(config.at("reference").at("style_bins")),
      style_channels_(config.at("acoustic").at("gin_channels")), type_(projection_.type()) {
    if (config.at("reference").at("sampling_rate") !=
            config.at("reference").at("mel_sample_rate") ||
        codebook_.cols != projection_.geometry().output_channels || codebook_.type != type_ ||
        style_bins_ < 1 || style_bins_ > spectrum_.output_channels())
        throw std::invalid_argument("inconsistent reference feature configuration");
}
ReferenceEncoder::Workspace::Workspace(const ReferenceEncoder &r, BufferAllocator *storage)
    : hubert(r.hubert_, 1, r.max_hubert_samples_, storage),
      style_encoder(r.style_, 1, r.spectrum_.frames(r.max_wave_samples_), storage),
      spectrum(r.spectrum_, storage), mel_spectrum(r.mel_, storage) {
    int frames = r.hubert_.frames(r.max_hubert_samples_);
    int projected_frames = r.projection_.geometry().output_length(frames);
    projected = Buffer(size_t(projected_frames) * r.codebook_.cols * bytes(r.type_), storage);
    convolution = Buffer(r.projection_.workspace_bytes(1, frames), storage);
    scores = Buffer(size_t(projected_frames) * r.codebook_.rows * bytes(r.type_), storage);
    codes = Buffer(size_t(projected_frames) * sizeof(int), storage);
    magnitude = Buffer(size_t(r.spectrum_.frames(r.max_wave_samples_)) *
                           r.spectrum_.output_channels() * bytes(r.type_),
                       storage);
    style = Buffer(size_t(r.style_channels_) * bytes(r.type_), storage);
    mel = Buffer(size_t(r.mel_.frames(r.max_wave_samples_)) * r.mel_.output_channels() *
                     bytes(r.type_),
                 storage);
}
ReferenceEncoder::Output ReferenceEncoder::forward(Workspace &w, TensorView wave,
                                                   TensorView hubert_wave) const {
    auto &gpu = backend_for(wave);
    if (wave.rows < 2 || wave.rows > max_wave_samples_ || wave.cols != 1 ||
        wave.type != Type::F32 || hubert_wave.rows < 1 || hubert_wave.rows > max_hubert_samples_ ||
        hubert_wave.cols != 1 || hubert_wave.type != type_)
        throw std::invalid_argument("reference waveform exceeds initialized capacity");
    auto features = hubert_.forward(w.hubert, hubert_wave, ForwardBatch{1, hubert_wave.rows});
    int time = projection_.geometry().output_length(features.rows);
    auto projected = w.projected.view(time, codebook_.cols, type_);
    auto scratch = w.convolution.view(
        features.rows, projection_.geometry().kernel * projection_.geometry().output_channels,
        Type::F32);
    projection_.forward(features, projected, ForwardBatch{1, features.rows}, scratch);
    auto scores = w.scores.view(time, codebook_.rows, type_);
    gpu.linear(projected, codebook_, scores, false, 0, 2);
    vector_quantize(gpu.stream, projected, scores, codebook_, norms_, w.codes.ptr<int>(), 0, 1,
                    false, true);
    int spec_frames = spectrum_.frames(wave.rows), mel_frames = mel_.frames(wave.rows);
    auto spectrum = w.magnitude.view(spec_frames, spectrum_.output_channels(), type_);
    spectrum_.forward(w.spectrum, wave, 1, wave.rows, spectrum);
    auto style = w.style.view(1, style_channels_, type_);
    style_.forward(w.style_encoder, spectrum.columns(0, style_bins_), style, ForwardBatch{1, spec_frames});
    auto mel = w.mel.view(mel_frames, mel_.output_channels(), type_);
    mel_.forward(w.mel_spectrum, wave, 1, wave.rows, mel);
    return {w.codes.view(time, 1, Type::I32), style, mel};
}
} // namespace tts::gpt_sovits
