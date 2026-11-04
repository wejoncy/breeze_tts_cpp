#include "models/gpt_sovits/model.h"
#include "io/json.h"
#include <algorithm>
#include <cmath>

namespace tts::gpt_sovits {
Model::Model(Weights &weights, const Json &configuration,
              const Json &pronunciation_configuration, int context)
    : config(configuration),
      bert(weights, config.at("bert"), "bert.bert",
           config.at("bert").at("num_hidden_layers").get<int>() - 2, weights.quantization() != Quantization::None),
      pronunciation(weights, "g2pw", pronunciation_configuration),
      semantic(weights, "semantic.model", config.at("semantic"), context),
      reference(weights, "hubert", "acoustic", config.at("hubert"), config.at("acoustic"),
                config.at("reference"), config.at("flow").at("mel_channels")),
      acoustic(weights, "acoustic", config.at("acoustic")),
      flow(weights, "acoustic.cfm.estimator", config.at("flow"), config.at("synthesis").at("window_frames"),
            config.at("reference").at("mel_min"), config.at("reference").at("mel_max"),
            config.at("synthesis").at("rolling_frames")),
      vocoder(weights, "vocoder", config.at("vocoder")),
      sampler({1, config.at("semantic").at("EOS"), config.at("semantic").at("EOS")}),
      window_frames(config.at("synthesis").at("window_frames")),
      reference_frames(config.at("synthesis").at("reference_frames")),
      rolling_frames(config.at("synthesis").at("rolling_frames")),
      target_frames(config.at("synthesis").at("target_frames")),
      mel_channels(config.at("flow").at("mel_channels")),
      feature_channels(config.at("flow").at("text_channels")),
      mel_minimum(config.at("reference").at("mel_min")),
      mel_maximum(config.at("reference").at("mel_max")) {
    if (config.at("model_type") != "gpt_sovits_v5" || window_frames < 2 || reference_frames < 1 ||
        reference_frames >= window_frames || rolling_frames < 0 ||
        rolling_frames > reference_frames || target_frames < 1 || target_frames > window_frames ||
        !(mel_maximum > mel_minimum) || !std::isfinite(mel_minimum) ||
        !std::isfinite(mel_maximum) || vocoder.samples(1) != config.at("samples_per_frame"))
        throw std::invalid_argument("inconsistent GPT-SoVITS synthesis configuration");
}
TensorView Model::text_features(TensorInputs inputs, ForwardContext &context) const {
    if (!inputs["input_ids"].numel()) return TensorView::metadata(0, semantic.hidden(), semantic.type());
    auto text_context = context;
    if (context.sequences.contains("text")) {
        const auto sequence = context.sequences["text"];
        text_context.batch = ForwardBatch(sequence);
        text_context.batch.positions = sequence.row_position;
        text_context.max_batch = std::max(1, sequence.sequences);
        text_context.max_time = std::max(1, sequence.max_length);
        text_context.max_rows = context.mode == ForwardContext::Prepare ? context.max_rows : sequence.rows;
    }
    auto hidden_states = bert.forward(inputs["input_ids"], text_context);
    return semantic.encode_features(hidden_states, text_context);
}
void Model::forward(std::string_view program, TensorInputs inputs, TensorOutputs outputs,
                     ForwardContext &context) const {
    if (program == "pronunciation") {
        outputs["labels"] = pronunciation.forward(inputs, context);
    } else if (program == "text") {
        outputs["encoded_features"] = text_features(inputs, context);
    } else if (program == "tokens") {
        std::vector<NamedTensor> arguments(inputs.data, inputs.data + inputs.size);
        if (!context.batch.is_decode()) arguments.push_back({"encoded_features", text_features(inputs, context)});
        auto logits = semantic.forward({arguments.data(), arguments.size()}, context);
        outputs["logits"] = logits;
        outputs["done"] = sampler.forward(logits, inputs["sampling"], inputs["done"], context);
    } else if (program == "waveform") {
        TensorBatch features;
        {
            Tensor::Scope acoustic_scope;
            features = acoustic.forward(inputs, context);
        }
        auto mel = flow.forward(features, inputs, context);
        outputs["waveform"] = vocoder.forward(mel, outputs["audio_layout"], context);
    } else if (program == "reference") {
        const auto encoded = reference.forward(inputs["waveform"], inputs["speech_waveform"], context);
        auto acoustic_context = context;
        acoustic_context.max_time = std::max(context.max_time, encoded.codes.rows);
        acoustic_context.max_rows = std::max(context.max_rows, inputs["phones"].rows);
        const auto conditioned = acoustic.forward(encoded.codes, inputs["phones"], encoded.style,
                                                   acoustic_context);
        outputs["codes"] = encoded.codes;
        outputs["style"] = encoded.style;
        outputs["mel"] = encoded.mel;
        outputs["features"] = conditioned.features;
        outputs["waveform_condition"] = conditioned.waveform_condition;
        outputs["reference_bert"] = text_features(inputs, context);
    } else {
        throw std::invalid_argument("unknown model forward program");
    }
}
} // namespace tts::gpt_sovits
