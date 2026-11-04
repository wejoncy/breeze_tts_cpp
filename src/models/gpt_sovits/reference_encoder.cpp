#include "ops/basic.h"
#include "models/gpt_sovits/reference_encoder.h"
#include "ops/vector_quantization.h"
#include <cmath>

namespace tts {
ReferenceEncoder::Output ReferenceEncoder::forward(TensorView waveform, TensorView speech_waveform,
                                                     ForwardContext &context) const {
    if (waveform.rows < 2 || waveform.rows > max_wave_samples_ || waveform.cols != 1 ||
        waveform.type != Type::F32 || speech_waveform.rows < 1 || speech_waveform.rows > max_hubert_samples_ ||
        speech_waveform.cols != 1 || speech_waveform.type != type_)
        throw std::invalid_argument("reference waveform exceeds initialized capacity");
    auto speech_context = context;
    speech_context.batch = ForwardBatch{1, speech_waveform.rows};
    auto features = hubert_.forward(speech_waveform, speech_context);
    const int time = projection_.geometry().output_length(features.rows);
    auto codes = context.tensor({time, 1}, Type::I32);
    {
        Tensor::Scope quantization;
        auto projected = context.tensor({time, codebook_.cols}, type_);
        auto scores = context.tensor({time, codebook_.rows}, type_);
        Buffer storage(projection_.workspace_bytes(1, features.rows));
        TensorView scratch;
        if (storage.size())
            scratch = storage.view(features.rows,
                projection_.geometry().kernel * projection_.geometry().output_channels, Type::F32);
        if (context.mode == ForwardContext::Prepare && projected.data) {
            const int maximum_frames = hubert_.frames(max_hubert_samples_);
            if (projection_.workspace_bytes(1, maximum_frames))
                projection_.prepare(features.slice(0), scratch.slice(0), maximum_frames);
            ops::prepare_matmul(projected.slice(0), codebook_, scores.slice(0), false, 0, 2, 1,
                projection_.geometry().output_length(maximum_frames));
        } else if (context.mode == ForwardContext::Execute) {
            projection_.forward(features, projected, ForwardBatch{1, features.rows}, scratch);
            ops::matmul(projected, codebook_, scores, false, 0, 2);
            vector_quantize(projected, scores, codebook_, norms_, codes.ptr<int>(), 0, 1, false, true);
        }
    }
    auto waveform_context = context;
    waveform_context.batch = ForwardBatch{1, waveform.rows};
    auto spectrum = spectrum_.forward(waveform, waveform_context);
    auto style_context = context;
    style_context.batch = ForwardBatch{1, spectrum.rows};
    auto style = style_.forward(spectrum.columns(0, style_bins_), style_context);
    auto mel = mel_.forward(waveform, waveform_context);
    return {codes, style, mel};
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
    std::vector<float> norms(table.rows, 0);
    for (int row_index = 0; row_index < table.rows; ++row_index) {
        double sum = 0;
        for (int channel_index = 0; channel_index < table.cols; ++channel_index) {
            float value = values[size_t(row_index) * table.cols + channel_index];
            sum += ops::rounded(value * value, table.type);
        }
        norms[row_index] = ops::rounded(float(sum), table.type);
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
                spectral_capacity(reference_config, 0), weights.device(), projection_.type()),
      mel_(spectral_config(reference_config, mel_channels),
           spectral_capacity(reference_config, mel_channels), weights.device(), projection_.type()),
      max_wave_samples_(int(reference_config.at("max_seconds").get<float>() *
                            reference_config.at("sampling_rate").get<int>())),
      max_hubert_samples_(int(reference_config.at("max_seconds").get<float>() *
                                  reference_config.at("hubert_sample_rate").get<int>() +
                              reference_config.at("semantic_silence_seconds").get<float>() *
                                  reference_config.at("sampling_rate").get<int>())),
      style_bins_(reference_config.at("style_bins")),
      type_(projection_.type()) {
    if (reference_config.at("sampling_rate") !=
            reference_config.at("mel_sample_rate") ||
        codebook_.cols != projection_.geometry().output_channels || codebook_.type != type_ ||
        style_bins_ < 1 || style_bins_ > spectrum_.output_channels())
        throw std::invalid_argument("inconsistent reference feature configuration");
}
} // namespace tts
