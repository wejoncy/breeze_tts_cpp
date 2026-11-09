#include "ops/basic.h"
#include "models/gpt_sovits/acoustic.h"
#include "layers/input_metadata.h"
#include "ops/attention.h"
namespace tts {
Acoustic::Output Acoustic::forward(TensorView codes, TensorView phones, TensorView style,
                                    ForwardContext &context) const {
    const int batch = context.batch.batch();
    if (batch < 1 || codes.type != Type::I32 || phones.type != Type::I32 ||
        codes.cols != 1 || phones.cols != 1 || codes.rows < 1 || phones.rows < 1 ||
        codes.rows % batch || phones.rows % batch)
        throw std::invalid_argument("acoustic token tensor shape/type mismatch");
    const auto layout = describe(batch, codes.rows / batch, phones.rows / batch);
    auto features = context.tensor({layout.features.rows, bridge_.output_channels()}, type_);
    auto condition = waveform_features_.condition(style, context);
    Tensor::Scope intermediate;
    forward(codes.ptr<int>(), phones.ptr<int>(), style, features, layout, context, condition);
    return {features, condition};
}
Acoustic::Acoustic(Weights &w, const std::string &prefix, const Json &cfg)
    : phones_(w.get(prefix + ".enc_p.text_embedding.weight")),
      ssl_encoder_(w, prefix + ".enc_p.encoder_ssl", cfg.at("n_layers").get<int>() / 2,
                   cfg.at("n_heads")),
      text_encoder_(w, prefix + ".enc_p.encoder_text", cfg.at("n_layers"), cfg.at("n_heads")),
      output_encoder_(w, prefix + ".enc_p.encoder2", cfg.at("n_layers").get<int>() / 2,
                      cfg.at("n_heads")),
      content_(w, prefix + ".enc_p.mrte.c_pre"), text_(w, prefix + ".enc_p.mrte.text_pre"),
      query_(w, prefix + ".enc_p.mrte.cross_attention.conv_q"),
      key_value_(
          w.fused(prefix + ".mrte.kv.weight", {prefix + ".enc_p.mrte.cross_attention.conv_k.weight",
                                              prefix + ".enc_p.mrte.cross_attention.conv_v.weight"}),
          w.fused(prefix + ".mrte.kv.bias", {prefix + ".enc_p.mrte.cross_attention.conv_k.bias",
                                            prefix + ".enc_p.mrte.cross_attention.conv_v.bias"})),
      attention_out_(w, prefix + ".enc_p.mrte.cross_attention.conv_o"),
      content_out_(w, prefix + ".enc_p.mrte.c_post"), bridge_(w, prefix + ".bridge.0"),
      waveform_features_(w, prefix + ".wns1", cfg.at("wn_layers"), cfg.at("wn_dilation_rate")),
      heads_(cfg.at("mrte_heads")), rate_(cfg.at("semantic_repeat"), cfg.at("feature_repeat")),
      type_(phones_.type) {
    for (const char *projection : {"conv_k", "conv_v"}) {
        auto name = std::string(prefix + ".enc_p.mrte.cross_attention.") + projection;
        w.validate(name + ".weight", {query_.output_channels(), text_.output_channels(), 1}, type_);
        w.validate(name + ".bias", {query_.output_channels()}, type_);
    }
    if (heads_ < 1 || content_.output_channels() % heads_ ||
        cfg.at("gin_channels").get<int>() != content_.output_channels())
        throw std::invalid_argument("invalid acoustic rates/attention geometry");
    codebook_ = w.get(prefix + ".quantizer.vq.layers.0._codebook.embed");
    codebook_projection_ = std::make_unique<Linear>(w, prefix + ".enc_p.ssl_proj");
    projected_codebook_ =
        Tensor({codebook_.rows, codebook_projection_->output_channels()}, type_, w.device());
}
void Acoustic::prepare_weights() const {
    codebook_projection_->forward(codebook_, projected_codebook_.view());
}
Acoustic::Batch Acoustic::describe(int batch, int codes, int phones, double speed) const {
    Batch input;
    input.phones = SequenceLayout::uniform(batch, phones);
    input.hidden = SequenceLayout::uniform(batch, hidden_frames(codes));
    input.resized = SequenceLayout::uniform(batch, hidden_frames(codes, speed));
    input.features = SequenceLayout::uniform(batch, feature_frames(codes, speed));
    input.resample = input.hidden.max_length != input.resized.max_length;
    return input;
}
TensorView Acoustic::encode_text(const int *phones, const Batch &input,
                                  ForwardContext &context) const {
    auto text_context = context.with_sequence(input.phones);
    const int rows = text_context.batch.rows();
    if (context.mode == ForwardContext::Execute && !phones && !input.phone_ids)
        throw std::invalid_argument("acoustic text requires phone IDs");
    auto condition = context.tensor({rows, text_condition_channels()}, type_);
    Tensor::Scope scratch;
    auto text = context.tensor({rows, text_.input_channels()}, type_);
    if (context.mode == ForwardContext::Execute) {
        ops::EmbeddingFusion embedding;
        embedding.sequence_length = rows;
        embedding.id_rows = input.phone_ids;
        ops::embedding(phones_, phones, text, embedding);
    }
    text_encoder_.forward(text, text_context);
    auto projected = text_.forward(text, text_context);
    ops::Epilogue routing;
    routing.output_rows = input.text_outputs;
    key_value_.forward(projected, condition, text_context, routing);
    return condition;
}
void Acoustic::forward(const int *codes, const int *phones, TensorView style,
                       TensorView features, const Batch &input, ForwardContext &context,
                       TensorView condition) const {
    auto hidden_context = context.with_sequence(input.hidden);
    const int batch = input.hidden.sequences, time = input.hidden.max_length, rows = input.hidden.rows;
    const int phone_count = input.phones.max_length, phone_rows = input.phones.rows;
    const bool execute = context.mode == ForwardContext::Execute;
    if (input.phones.sequences != batch || input.resized.sequences != batch ||
        input.features.sequences != batch || phone_count < 1 || phone_rows < 1 ||
        (!input.styles && (style.rows != batch || style.cols != content_.output_channels() || style.type != type_)) ||
        features.rows != input.features.rows || features.cols != bridge_.output_channels() || features.type != type_ ||
        (execute && !codes && !input.code_ids) ||
        (input.text_keys && (!input.text_values || !input.cross_attention)))
        throw std::invalid_argument("invalid acoustic batch geometry");
    Tensor::Scope intermediate;
    auto ssl = context.tensor({rows, content_.input_channels()}, type_);
    if (execute) {
        ops::EmbeddingFusion embedding;
        embedding.sequence_length = rows;
        embedding.id_rows = input.code_ids;
        embedding.input_repeat = rate_.semantic_repeat;
        ops::embedding(projected_codebook_.view(), codes, ssl, embedding);
    }
    ssl_encoder_.forward(ssl, hidden_context);
    {
        Tensor::Scope attention_scratch;
        auto text_condition = input.text_keys
            ? context.tensor({phone_rows, text_condition_channels()}, type_)
            : encode_text(phones, input, context);
        auto content = content_.forward(ssl, hidden_context);
        auto query = query_.forward(content, hidden_context);
        auto attended = context.tensor({rows, query.cols}, type_);
        auto lse = context.tensor({rows, heads_}, Type::F32);
        if (execute) {
            DenseAttentionLayout attention{0, 0, input.cross_attention, input.text_keys, input.text_values};
            if (!input.cross_attention) {
                attention.cu_seqlens_q = input.hidden.cu_seqlens;
                attention.cu_seqlens_k = input.phones.cu_seqlens;
            }
            dense_attention(query, text_condition.columns(0, query.cols),
                text_condition.columns(query.cols, query.cols), attended, lse.ptr<float>(),
                batch, time, phone_count, heads_, query.cols / heads_, false, 0, attention);
        }
        auto update = context.tensor({rows, attention_out_.output_channels()}, type_);
        attention_out_.project(attended, update, hidden_context);
        if (execute) {
            ops::Epilogue fusion;
            fusion.bias = attention_out_.bias();
            fusion.residual = content;
            fusion.condition = style;
            fusion.condition_sources = input.styles;
            fusion.packed = input.hidden.addressing();
            fusion.rows_per_sequence = time;
            ops::pointwise(update, content, fusion);
        }
        content_out_.forward(content, ssl, hidden_context);
    }
    output_encoder_.forward(ssl, hidden_context);
    auto resized_context = context.with_sequence(input.resized);
    if (input.resample) {
        auto resized = context.tensor({input.resized.rows, ssl.cols}, type_);
        if (execute)
            ops::linear_resample(ssl, resized, batch, time, input.resized.max_length,
                                 input.hidden.addressing(), input.resized.addressing());
        ssl = resized;
    }
    auto update = context.tensor({input.resized.rows, bridge_.output_channels()}, type_);
    bridge_.project(ssl, update, resized_context);
    if (execute) {
        ops::Epilogue activation;
        activation.bias = bridge_.bias();
        activation.activation = ops::Activation::LeakyRelu;
        activation.alpha = .01f;
        activation.input_repeat = rate_.feature_repeat;
        ops::pointwise(update, features, activation);
    }
    auto feature_context = context.with_sequence(input.features);
    waveform_features_.forward(features, input.conditions.numel() ? input.conditions : condition,
        features, feature_context, input.feature_outputs);
}
TensorBatch Acoustic::forward(TensorInputs inputs, ForwardContext &context) const {
    TensorBatch result;
    result.rows = context.max_batch;
    const int max_codes = context.max_rows, max_phones = context.max_rows;
    const int max_frames = feature_capacity(max_codes);
    if (context.mode != ForwardContext::Execute) {
        for (int rows : {max_phones, hidden_frames(max_codes), max_frames, max_frames})
            Tensor::reserve((size_t(rows) * 2 + size_t(context.max_batch) * 2 + 1) * sizeof(int));
        Tensor::reserve(size_t(context.max_batch) * sizeof(AttentionSequence));
        Tensor::reserve(size_t(max_codes + max_phones + max_frames) * sizeof(void *));
        Tensor::reserve(size_t(context.max_batch) * 4 * sizeof(void *));
        auto layout = describe(1, max_codes, max_phones);
        layout.resized = SequenceLayout::uniform(1, max_frames / rate_.feature_repeat);
        layout.features = SequenceLayout::uniform(1, max_frames);
        layout.conditions = TensorView::metadata(1, 1, Type::I64);
        layout.resample = layout.hidden.rows != layout.resized.rows;
        auto features = context.tensor({max_frames, bridge_.output_channels()}, type_);
        forward(nullptr, nullptr, TensorView::metadata(1, content_.output_channels(), type_),
                features, layout, context);
        return result;
    }
    const auto generated = inputs["generated"];
    const auto phones = inputs["phone_lengths"];
    const auto speeds = inputs["speed"];
    const int batch = generated.rows;
    if (generated.type != Type::I32 || phones.type != Type::I32 || speeds.type != Type::F64 ||
        phones.rows != batch || speeds.rows != batch || batch > context.max_batch)
        throw std::invalid_argument("acoustic sequence metadata shape/type mismatch");
    const auto state_codes = context.state["codes"], state_phones = context.state["phones"];
    const auto styles = context.state["style"], conditions = context.state["waveform_condition"];
    const auto text_states = context.state["text_condition"], features = context.state["features"];
    const auto text_ready = context.state["text_ready"], feature_tokens = context.state["feature_tokens"];
    auto upload = [&](const void *data, int rows, int columns, Type type) {
        return upload_metadata(context, data, {rows, columns}, type);
    };
    for (int first = 0; first < batch;) {
        SequenceBuilder phone_layout;
        phone_layout.reserve(context.max_batch, max_phones);
        std::vector<const int *> phone_ids;
        std::vector<void *> destinations;
        std::vector<int> members;
        int next = first;
        while (next < batch) {
            if (*text_ready[next].ptr<int>()) { ++next; continue; }
            const int count = phones.ptr<int>()[next];
            if (count < 1 || count > max_phones)
                throw std::invalid_argument("acoustic text exceeds the declared token capacity");
            if (count > max_phones - phone_layout.rows() && !members.empty()) break;
            phone_layout.append(count, count);
            for (int row = 0; row < count; ++row) {
                phone_ids.push_back(state_phones[next].ptr<int>() + row);
                destinations.push_back(text_states[next].ptr<uint16_t>() + size_t(row) * text_condition_channels());
            }
            members.push_back(next++);
        }
        if (!members.empty()) {
            Tensor::Scope descriptors;
            Batch layout;
            layout.phones = upload_sequence(context, phone_layout);
            layout.phone_ids = upload(phone_ids.data(), int(phone_ids.size()), 1, Type::I64).ptr<const int *>();
            layout.text_outputs = upload(destinations.data(), int(destinations.size()), 1, Type::I64).ptr<void *>();
            encode_text(nullptr, layout, context);
            for (int member : members) *text_ready[member].ptr<int>() = 1;
        }
        first = next;
    }
    for (int first = 0; first < batch;) {
        SequenceBuilder phone_layout, hidden_layout, resized_layout, feature_layout;
        phone_layout.reserve(context.max_batch, max_phones);
        hidden_layout.reserve(context.max_batch, hidden_frames(max_codes));
        resized_layout.reserve(context.max_batch, max_frames);
        feature_layout.reserve(context.max_batch, max_frames);
        std::vector<AttentionSequence> attention;
        std::vector<const int *> code_ids;
        std::vector<const void *> style_rows, condition_rows, keys, values;
        std::vector<void *> destinations;
        std::vector<int> members;
        int code_count = 0, next = first;
        while (next < batch) {
            const int count = generated.ptr<int>()[next];
            if (*feature_tokens[next].ptr<int>() == count) { ++next; continue; }
            const int phone_count = phones.ptr<int>()[next];
            const double speed = speeds.ptr<double>()[next];
            const int frame_count = feature_frames(count, speed);
            const bool fits = count <= max_codes - code_count &&
                phone_count <= max_phones - phone_layout.rows() &&
                frame_count <= max_frames - feature_layout.rows();
            if (!fits) {
                if (members.empty()) throw std::invalid_argument("acoustic sequence exceeds its workspace capacity");
                break;
            }
            const int hidden = hidden_frames(count), resized = hidden_frames(count, speed);
            attention.push_back({hidden_layout.rows(), 0, hidden, phone_count});
            phone_layout.append(phone_count, phone_count);
            hidden_layout.append(hidden, hidden);
            resized_layout.append(resized, resized);
            feature_layout.append(frame_count, valid_frames(count, speed));
            for (int row = 0; row < count; ++row) code_ids.push_back(state_codes[next].ptr<int>() + row);
            style_rows.push_back(styles[next].data); condition_rows.push_back(conditions[next].data);
            keys.push_back(text_states[next].data);
            values.push_back(text_states[next].ptr<uint16_t>() + text_condition_channels() / 2);
            for (int row = 0; row < frame_count; ++row)
                destinations.push_back(features[next].ptr<uint16_t>() + size_t(row) * features[next].cols);
            members.push_back(next++);
            code_count += count;
        }
        if (!members.empty()) {
            Tensor::Scope descriptors;
            Batch layout;
            layout.phones = upload_sequence(context, phone_layout);
            layout.hidden = upload_sequence(context, hidden_layout);
            layout.resized = upload_sequence(context, resized_layout);
            layout.features = upload_sequence(context, feature_layout);
            layout.cross_attention = upload(attention.data(), int(attention.size()), sizeof(AttentionSequence), Type::I8).ptr<AttentionSequence>();
            layout.code_ids = upload(code_ids.data(), int(code_ids.size()), 1, Type::I64).ptr<const int *>();
            layout.styles = upload(style_rows.data(), int(style_rows.size()), 1, Type::I64).ptr<const void *>();
            layout.conditions = upload(condition_rows.data(), int(condition_rows.size()), 1, Type::I64);
            layout.text_keys = upload(keys.data(), int(keys.size()), 1, Type::I64).ptr<const void *>();
            layout.text_values = upload(values.data(), int(values.size()), 1, Type::I64).ptr<const void *>();
            layout.feature_outputs = upload(destinations.data(), int(destinations.size()), 1, Type::I64);
            layout.resample = !hidden_layout.same_extents(resized_layout);
            auto output = context.tensor({feature_layout.rows(), features[0].cols}, type_);
            forward(nullptr, nullptr, {}, output, layout, context);
            for (int member : members) *feature_tokens[member].ptr<int>() = generated.ptr<int>()[member];
        }
        first = next;
    }
    result.rows = batch;
    for (int row = 0; row < batch; ++row)
        result.chunks.push_back({features[row].slice(0,
            feature_frames(generated.ptr<int>()[row], speeds.ptr<double>()[row])), {row},
            feature_frames(generated.ptr<int>()[row], speeds.ptr<double>()[row])});
    return result;
}
} // namespace tts
