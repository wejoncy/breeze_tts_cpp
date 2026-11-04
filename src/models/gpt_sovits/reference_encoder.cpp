#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "ops/basic.h"
#include "models/gpt_sovits/reference_encoder.h"
#include "ops/vector_quantization.h"
#include <cmath>

namespace tts {
ReferenceEncoder::Output ReferenceEncoder::forward(TensorView waveform, TensorView speech_waveform,
                                                     ForwardContext &context) const {
    Workspace workspace(*this);
    if (context.mode == ForwardContext::Execute)
        return forward(workspace, waveform, speech_waveform);
    if (context.mode == ForwardContext::Prepare && workspace.projected.data()) {
        hubert_.prepare(workspace.hubert, max_hubert_samples_);
        const int frames = hubert_.frames(max_hubert_samples_);
        const int codes = projection_.geometry().output_length(frames);
        if (projection_.workspace_bytes(1, frames))
            projection_.prepare(workspace.hubert.projected.view(1, projection_.geometry().input_channels, type_),
                workspace.convolution.view(1, projection_.geometry().kernel * projection_.geometry().output_channels,
                                            Type::F32), frames);
        ops::prepare_matmul(workspace.projected.view(1, codebook_.cols, type_), codebook_,
            workspace.scores.view(1, codebook_.rows, type_), false, 0, 2, 1, codes);
        style_.prepare(workspace.style_encoder, workspace.magnitude.view(1, spectrum_.output_channels(), type_)
                                                  .columns(0, style_bins_));
        mel_.prepare(workspace.mel_spectrum);
    }
    const int codes = projection_.geometry().output_length(hubert_.frames(speech_waveform.rows));
    return {workspace.codes.view(codes, 1, Type::I32),
            workspace.style.view(1, style_channels_, type_),
            workspace.mel.view(mel_.frames(waveform.rows), mel_.output_channels(), type_)};
}
namespace {
SpectrogramConfig spectral_config(const Json &configuration, int channels) {
    int fft = configuration.at(channels ? "mel_fft" : "filter_length");
    int hop = configuration.at(channels ? "mel_hop" : "hop_length");
    int window = configuration.at(channels ? "mel_window" : "win_length");
    SpectrogramConfig result{fft, hop, window, (fft - hop) / 2,
                             configuration.at(channels ? "mel_sample_rate" : "sampling_rate")};
    result.mel_channels = channels;
    if (channels) {
        float low = configuration.at("mel_min"), high = configuration.at("mel_max");
        if (!(high > low))
            throw std::invalid_argument("invalid mel normalization range");
        result.log_scale = 2.f / (high - low);
        result.log_bias = -low * result.log_scale - 1;
    }
    return result;
}
int spectral_capacity(const Json &configuration, int channels) {
    auto geometry = spectral_config(configuration, channels);
    return geometry.frames(int(configuration.at("max_seconds").get<float>() * geometry.sample_rate));
}
ops::ConvGeometry projection_geometry(Weights &weights, const std::string &prefix, const Json &configuration) {
    const auto &shape = weights.entry(prefix + ".ssl_proj.weight").shape;
    int stride = configuration.at("semantic_frame_rate") == "25hz" ? 2 : 1;
    if (shape.size() != 3 || shape[2] != stride)
        throw std::invalid_argument("semantic feature projection stride mismatch");
    return {shape[1], shape[0], shape[2], stride};
}
TensorView codebook_norms(Weights &weights, const std::string &prefix) {
    const std::string name = prefix + ".quantizer.vq.layers.0._codebook.embed";
    auto table = weights.get(name);
    auto values = weights.host_values(name);
    auto round = [&](float value) {
        if (table.type == Type::F16)
            return float(__float2half_rn(value));
        if (table.type == Type::BF16)
            return float(__float2bfloat16_rn(value));
        return value;
    };
    std::vector<float> norms(table.rows, 0);
    for (int row_index = 0; row_index < table.rows; ++row_index) {
        double sum = 0;
        for (int channel_index = 0; channel_index < table.cols; ++channel_index) {
            float value = values[size_t(row_index) * table.cols + channel_index];
            sum += round(value * value);
        }
        norms[row_index] = round(float(sum));
    }
    return weights.generated(name + ".norms", norms, 1, table.rows, table.type);
}
} // namespace
ReferenceEncoder::ReferenceEncoder(Weights &weights, const std::string &speech_prefix,
                                     const std::string &acoustic_prefix, const Json &speech_config,
                                     const Json &acoustic_config, const Json &reference_config,
                                     int mel_channels)
    : hubert_(weights, speech_prefix, speech_config),
      projection_(weights, acoustic_prefix + ".ssl_proj", projection_geometry(weights, acoustic_prefix, acoustic_config)),
      codebook_(weights.get(acoustic_prefix + ".quantizer.vq.layers.0._codebook.embed")), norms_(codebook_norms(weights, acoustic_prefix)),
      style_(weights, acoustic_prefix + ".ref_enc", acoustic_config.at("style_heads"),
             acoustic_config.at("style_temporal_layers")),
      spectrum_(spectral_config(reference_config, 0),
                spectral_capacity(reference_config, 0), weights.device()),
      mel_(spectral_config(reference_config, mel_channels),
           spectral_capacity(reference_config, mel_channels), weights.device()),
      max_wave_samples_(int(reference_config.at("max_seconds").get<float>() *
                            reference_config.at("sampling_rate").get<int>())),
      max_hubert_samples_(int(reference_config.at("max_seconds").get<float>() *
                                  reference_config.at("hubert_sample_rate").get<int>() +
                              reference_config.at("semantic_silence_seconds").get<float>() *
                                  reference_config.at("sampling_rate").get<int>())),
      style_bins_(reference_config.at("style_bins")),
      style_channels_(acoustic_config.at("gin_channels")), type_(projection_.type()) {
    if (reference_config.at("sampling_rate") !=
            reference_config.at("mel_sample_rate") ||
        codebook_.cols != projection_.geometry().output_channels || codebook_.type != type_ ||
        style_bins_ < 1 || style_bins_ > spectrum_.output_channels())
        throw std::invalid_argument("inconsistent reference feature configuration");
}
ReferenceEncoder::Workspace::Workspace(const ReferenceEncoder &encoder)
    : hubert(encoder.hubert_, 1, encoder.max_hubert_samples_),
      style_encoder(encoder.style_, 1, encoder.spectrum_.frames(encoder.max_wave_samples_)),
      spectrum(encoder.spectrum_), mel_spectrum(encoder.mel_) {
    int frames = encoder.hubert_.frames(encoder.max_hubert_samples_);
    int projected_frames = encoder.projection_.geometry().output_length(frames);
    projected = Buffer(size_t(projected_frames) * encoder.codebook_.cols * bytes(encoder.type_));
    convolution = Buffer(encoder.projection_.workspace_bytes(1, frames));
    scores = Buffer(size_t(projected_frames) * encoder.codebook_.rows * bytes(encoder.type_));
    codes = Buffer(size_t(projected_frames) * sizeof(int));
    magnitude = Buffer(size_t(encoder.spectrum_.frames(encoder.max_wave_samples_)) *
                           encoder.spectrum_.output_channels() * bytes(encoder.type_));
    style = Buffer(size_t(encoder.style_channels_) * bytes(encoder.type_));
    mel = Buffer(size_t(encoder.mel_.frames(encoder.max_wave_samples_)) * encoder.mel_.output_channels() *
                     bytes(encoder.type_));
}
ReferenceEncoder::Output ReferenceEncoder::forward(Workspace &workspace, TensorView waveform,
                                                   TensorView speech_waveform) const {
    if (waveform.rows < 2 || waveform.rows > max_wave_samples_ || waveform.cols != 1 ||
        waveform.type != Type::F32 || speech_waveform.rows < 1 || speech_waveform.rows > max_hubert_samples_ ||
        speech_waveform.cols != 1 || speech_waveform.type != type_)
        throw std::invalid_argument("reference waveform exceeds initialized capacity");
    auto features = hubert_.forward(workspace.hubert, speech_waveform, ForwardBatch{1, speech_waveform.rows});
    int time = projection_.geometry().output_length(features.rows);
    auto projected = workspace.projected.view(time, codebook_.cols, type_);
    auto scratch = workspace.convolution.view(
        features.rows, projection_.geometry().kernel * projection_.geometry().output_channels,
        Type::F32);
    projection_.forward(features, projected, ForwardBatch{1, features.rows}, scratch);
    auto scores = workspace.scores.view(time, codebook_.rows, type_);
    ops::matmul(projected, codebook_, scores, false, 0, 2);
    vector_quantize( projected, scores, codebook_, norms_, workspace.codes.ptr<int>(), 0, 1,
                    false, true);
    int spec_frames = spectrum_.frames(waveform.rows), mel_frames = mel_.frames(waveform.rows);
    auto spectrum = workspace.magnitude.view(spec_frames, spectrum_.output_channels(), type_);
    spectrum_.forward(workspace.spectrum, waveform, 1, waveform.rows, spectrum);
    auto style = workspace.style.view(1, style_channels_, type_);
    style_.forward(workspace.style_encoder, spectrum.columns(0, style_bins_), style, ForwardBatch{1, spec_frames});
    auto mel = workspace.mel.view(mel_frames, mel_.output_channels(), type_);
    mel_.forward(workspace.mel_spectrum, waveform, 1, waveform.rows, mel);
    return {workspace.codes.view(time, 1, Type::I32), style, mel};
}
} // namespace tts
