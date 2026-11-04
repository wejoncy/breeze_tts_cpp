#include "ops/basic.h"
#include "layers/audio/acoustic.h"
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
    Workspace workspace(*this, context.max_batch, context.max_time, context.max_rows,
                         &context.storage);
    const auto layout = describe(batch, codes.rows / batch, phones.rows / batch);
    auto features = workspace.features.view(layout.features.rows, bridge_.output_channels(), type_);
    if (context.mode == ForwardContext::Prepare && workspace.ssl.data())
        prepare(workspace);
    else if (context.mode == ForwardContext::Execute)
        forward(workspace, codes.ptr<int>(), phones.ptr<int>(), style, features, layout);
    return {features, workspace.waveform_features.condition.view(
                          batch, waveform_features_.condition_channels(), type_)};
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
    auto codebook = w.get(prefix + ".quantizer.vq.layers.0._codebook.embed");
    Linear project(w, prefix + ".enc_p.ssl_proj");
    projected_codebook_ =
        Tensor({codebook.rows, project.output_channels()}, type_, &w.allocator());
    w.after_loading([project, codebook, output = projected_codebook_.view()] {
        project.forward(codebook, output);
    });
}
namespace {
int acoustic_rows(int batch, int time) {
    if (batch < 1 || time < 1 || int64_t(batch) * time > INT32_MAX)
        throw std::invalid_argument("invalid acoustic capacity");
    return batch * time;
}
}
Acoustic::Capacity Acoustic::checked_capacity(Capacity capacity, int repeat) {
    if (capacity.sequences < 1 || capacity.codes < 1 || capacity.phones < 1 ||
        capacity.frames < 1 || capacity.frames % repeat)
        throw std::invalid_argument("invalid acoustic capacity");
    return capacity;
}
Acoustic::Workspace::Workspace(const Acoustic &a, int batch, int codes, int phones,
                               BufferAllocator *allocator, int frames)
    : Workspace(a, Capacity{batch, acoustic_rows(batch, codes), acoustic_rows(batch, phones),
                            acoustic_rows(batch, frames ? frames : a.feature_frames(codes))}, allocator) {}
Acoustic::Workspace::Workspace(const Acoustic &a, Capacity cap, BufferAllocator *allocator)
    : ssl_encoder(a.ssl_encoder_, 1, a.hidden_frames(checked_capacity(cap, a.rate_.feature_repeat).codes), allocator),
      text_encoder(a.text_encoder_, 1, cap.phones, allocator),
      output_encoder(a.output_encoder_, 1, a.hidden_frames(cap.codes), allocator),
      waveform_features(a.waveform_features_, 1, cap.frames, allocator, cap.sequences),
      max_batch(cap.sequences), max_codes(cap.codes), max_phones(cap.phones), max_frames(cap.frames) {
    ssl_encoder.max_batch = text_encoder.max_batch = output_encoder.max_batch = cap.sequences;
    size_t rows = a.hidden_frames(cap.codes), phone_rows = cap.phones;
    size_t resized_rows = cap.frames / a.rate_.feature_repeat;
    auto tensor = [&](size_t n, int width) {
        return Buffer(n * width * bytes(a.type_), allocator);
    };
    ssl = tensor(rows, a.content_.input_channels());
    text = tensor(phone_rows, a.text_.input_channels());
    content = Buffer(std::max(rows * a.content_.output_channels(),
                              resized_rows * a.content_out_.output_channels()) * bytes(a.type_), allocator);
    projected_text = tensor(phone_rows, a.text_.output_channels());
    query = tensor(rows, a.query_.output_channels());
    key_value = tensor(phone_rows, a.key_value_.output_channels());
    attended = tensor(rows, a.query_.output_channels());
    update = tensor(std::max(rows, resized_rows), a.attention_out_.output_channels());
    if (a.bridge_.output_channels() != a.attention_out_.output_channels())
        throw std::invalid_argument("acoustic bridge scratch width mismatch");
    features = tensor(cap.frames, a.bridge_.output_channels());
    lse = Buffer(rows * a.heads_ * sizeof(float), allocator);
}
void Acoustic::prepare(Workspace &w) const {
    int rows = hidden_frames(w.max_codes), resized = w.max_frames / rate_.feature_repeat;
    ssl_encoder_.prepare(w.ssl_encoder, rows);
    text_encoder_.prepare(w.text_encoder, w.max_phones);
    output_encoder_.prepare(w.output_encoder, rows);
    auto plan = [&](const Linear &l, Buffer &input, Buffer &output, int maximum) {
        l.prepare_range(input.view(1, l.input_channels(), type_),
                         output.view(1, l.output_channels(), type_), maximum);
    };
    plan(content_, w.ssl, w.content, rows);
    plan(query_, w.content, w.query, rows);
    plan(text_, w.text, w.projected_text, w.max_phones);
    plan(key_value_, w.projected_text, w.key_value, w.max_phones);
    plan(attention_out_, w.attended, w.update, rows);
    plan(content_out_, w.content, w.ssl, rows);
    plan(bridge_, w.content, w.update, resized);
    waveform_features_.prepare(w.waveform_features, w.max_frames);
}
Acoustic::Batch Acoustic::describe(int batch, int codes, int phones, double speed) const {
    Batch input;
    input.phones = SequenceLayout::uniform(batch, phones);
    input.hidden = SequenceLayout::uniform(batch, hidden_frames(codes));
    input.resized = SequenceLayout::uniform(batch, hidden_frames(codes, speed));
    input.features = SequenceLayout::uniform(batch, feature_frames(codes, speed));
    input.resample = input.hidden.max_length != input.resized.max_length;
    input.valid_frames = valid_frames(codes, speed);
    return input;
}
void Acoustic::prepare_text(Workspace &w, const int *phones, TensorView condition,
                            const Batch &input) const {

    ForwardBatch text_batch(input.phones);
    int rows = text_batch.rows();
    if ((!phones && !input.phone_ids) || !condition.data ||
        text_batch.batch() > w.max_batch || rows > w.max_phones || condition.rows != rows ||
        condition.cols != text_condition_channels() || condition.type != type_)
        throw std::invalid_argument("acoustic text exceeds initialized workspace");
    auto text = w.text.view(rows, text_.input_channels(), type_);
    ops::EmbeddingFusion f;
    f.sequence_length = rows;
    f.id_rows = input.phone_ids;
    ops::embedding( phones_, phones, text, f);
    text_encoder_.forward(w.text_encoder, text, text_batch);
    auto projected = w.projected_text.view(rows, text_.output_channels(), type_);
    text_.forward(text, projected);
    ops::Epilogue e;
    e.output_rows = input.text_outputs;
    key_value_.forward(projected, condition, e);
}
void Acoustic::forward(Workspace &w, const int *codes, const int *phones,
                       TensorView style, TensorView features, const Batch &input,
                       TensorView text_condition) const {

    ForwardBatch hidden_batch(input.hidden), feature_batch(input.features);
    int batch = hidden_batch.batch(), time = hidden_batch.time(), rows = hidden_batch.rows();
    int phone_count = input.phones.max_length, phone_rows = input.phones.rows;
    int feature_rows = feature_batch.rows();
    if ((!codes && !input.code_ids) || batch > w.max_batch ||
        rows % rate_.semantic_repeat || rows / rate_.semantic_repeat > w.max_codes ||
        input.phones.sequences != batch || input.resized.sequences != batch ||
        input.features.sequences != batch || phone_count < 1 || phone_rows < 1 ||
        phone_rows > w.max_phones || (!input.styles &&
        (style.rows != batch || style.cols != content_.output_channels() || style.type != type_)) ||
        feature_rows > w.max_frames || features.rows != feature_rows ||
        features.cols != bridge_.output_channels() || features.type != type_ ||
        (text_condition.data && (text_condition.rows != phone_rows ||
         text_condition.cols != text_condition_channels() || text_condition.type != type_)) ||
        (input.text_keys && (!input.text_values || !input.cross_attention)))
        throw std::invalid_argument("acoustic input exceeds initialized workspace");
    auto ssl = w.ssl.view(rows, content_.input_channels(), type_);
    ops::EmbeddingFusion f;
    f.sequence_length = rows;
    f.id_rows = input.code_ids;
    f.input_repeat = rate_.semantic_repeat;
    ops::embedding( projected_codebook_.view(), codes, ssl, f);
    ssl_encoder_.forward(w.ssl_encoder, ssl, hidden_batch);
    if (!text_condition.data) {
        text_condition = w.key_value.view(phone_rows, text_condition_channels(), type_);
        if (!input.text_keys) prepare_text(w, phones, text_condition, input);
    }
    auto content = w.content.view(rows, content_.output_channels(), type_);
    content_.forward(ssl, content);
    auto query = w.query.view(rows, query_.output_channels(), type_);
    query_.forward(content, query);
    auto attended = w.attended.view(rows, query.cols, type_);
    DenseAttentionLayout attention{0, 0, input.cross_attention, input.text_keys, input.text_values};
    if (!input.cross_attention) {
        attention.cu_seqlens_q = input.hidden.cu_seqlens;
        attention.cu_seqlens_k = input.phones.cu_seqlens;
    }
    dense_attention( query, text_condition.columns(0, query.cols),
                    text_condition.columns(query.cols, query.cols), attended, w.lse.ptr<float>(),
                    batch, time, phone_count, heads_, query.cols / heads_, false, 0, attention);
    auto update = w.update.view(rows, attention_out_.output_channels(), type_);
    attention_out_.project(attended, update);
    ops::Epilogue e;
    e.bias = attention_out_.bias();
    e.residual = content;
    e.condition = style;
    e.condition_sources = input.styles;
    e.packed = input.hidden.addressing();
    e.rows_per_sequence = time;
    ops::pointwise( update, content, e);
    content_out_.forward(content, ssl);
    output_encoder_.forward(w.output_encoder, ssl, hidden_batch);
    int resized_time = input.resized.max_length, resized_rows = input.resized.rows;
    if (input.resample) {
        auto resized = w.content.view(resized_rows, ssl.cols, type_);
        ops::linear_resample( ssl, resized, batch, time, resized_time,
                             input.hidden.addressing(), input.resized.addressing());
        ssl = resized;
    }
    update = w.update.view(resized_rows, bridge_.output_channels(), type_);
    bridge_.project(ssl, update);
    e = {};
    e.bias = bridge_.bias();
    e.activation = ops::Activation::LeakyRelu;
    e.alpha = .01f;
    e.input_repeat = rate_.feature_repeat;
    ops::pointwise( update, features, e);
    waveform_features_.forward(w.waveform_features, features, style, features, feature_batch,
                               input.valid_frames, input.conditions, input.feature_outputs);
}
} // namespace tts

namespace tts {
TensorBatch Acoustic::forward(TensorInputs inputs, ForwardContext &context) const {
    TensorBatch result;
    result.rows = context.max_batch;
    Workspace workspace(*this, Capacity{context.max_batch, context.max_rows, context.max_rows,
                                        feature_capacity(context.max_rows)}, &context.storage);
    if (context.mode == ForwardContext::Prepare && workspace.ssl.data()) prepare(workspace);
    if (context.mode != ForwardContext::Execute) {
        WorkspacePlan metadata;
        for (int rows : {workspace.max_phones, hidden_frames(workspace.max_codes),
                         workspace.max_frames, workspace.max_frames})
            metadata.allocate((size_t(rows) * 2 + size_t(context.max_batch) * 2 + 1) * sizeof(int));
        metadata.allocate(size_t(context.max_batch) * sizeof(AttentionSequence));
        metadata.allocate(size_t(workspace.max_codes + workspace.max_phones +
                                 workspace.max_frames) * sizeof(void *));
        metadata.allocate(size_t(context.max_batch) * 4 * sizeof(void *));
        context.storage.allocate(metadata.size);
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
        phone_layout.reserve(context.max_batch, workspace.max_phones);
        std::vector<const int *> phone_ids;
        std::vector<void *> destinations;
        std::vector<int> members;
        int next = first;
        while (next < batch) {
            if (*text_ready[next].ptr<int>()) { ++next; continue; }
            const int count = phones.ptr<int>()[next];
            if (count < 1 || count > workspace.max_phones)
                throw std::invalid_argument("acoustic text exceeds the declared token capacity");
            if (count > workspace.max_phones - phone_layout.rows() && !members.empty()) break;
            phone_layout.append(count, count);
            for (int row = 0; row < count; ++row) {
                phone_ids.push_back(state_phones[next].ptr<int>() + row);
                destinations.push_back(text_states[next].ptr<uint16_t>() + size_t(row) * text_condition_channels());
            }
            members.push_back(next++);
        }
        if (!members.empty()) {
            ScratchScope descriptors(context.storage);
            Batch layout;
            layout.phones = upload_sequence(context, phone_layout);
            layout.phone_ids = upload(phone_ids.data(), int(phone_ids.size()), 1, Type::I64).ptr<const int *>();
            layout.text_outputs = upload(destinations.data(), int(destinations.size()), 1, Type::I64).ptr<void *>();
            prepare_text(workspace, nullptr,
                workspace.key_value.view(phone_layout.rows(), text_condition_channels(), type_), layout);
            for (int member : members) *text_ready[member].ptr<int>() = 1;
        }
        first = next;
    }
    for (int first = 0; first < batch;) {
        SequenceBuilder phone_layout, hidden_layout, resized_layout, feature_layout;
        phone_layout.reserve(context.max_batch, workspace.max_phones);
        hidden_layout.reserve(context.max_batch, hidden_frames(workspace.max_codes));
        resized_layout.reserve(context.max_batch, workspace.max_frames);
        feature_layout.reserve(context.max_batch, workspace.max_frames);
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
            const bool fits = count <= workspace.max_codes - code_count &&
                phone_count <= workspace.max_phones - phone_layout.rows() &&
                frame_count <= workspace.max_frames - feature_layout.rows();
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
            ScratchScope descriptors(context.storage);
            Batch layout;
            layout.phones = upload_sequence(context, phone_layout);
            layout.hidden = upload_sequence(context, hidden_layout);
            layout.resized = upload_sequence(context, resized_layout);
            layout.features = upload_sequence(context, feature_layout);
            layout.cross_attention = upload(attention.data(), int(attention.size()), sizeof(AttentionSequence), Type::I8).ptr<AttentionSequence>();
            layout.code_ids = upload(code_ids.data(), int(code_ids.size()), 1, Type::I64).ptr<const int *>();
            layout.styles = upload(style_rows.data(), int(style_rows.size()), 1, Type::I64).ptr<const void *>();
            layout.conditions = upload(condition_rows.data(), int(condition_rows.size()), 1, Type::I64).ptr<const void *>();
            layout.text_keys = upload(keys.data(), int(keys.size()), 1, Type::I64).ptr<const void *>();
            layout.text_values = upload(values.data(), int(values.size()), 1, Type::I64).ptr<const void *>();
            layout.feature_outputs = upload(destinations.data(), int(destinations.size()), 1, Type::I64).ptr<void *>();
            layout.resample = !hidden_layout.same_extents(resized_layout);
            forward(workspace, nullptr, nullptr, {},
                workspace.features.view(feature_layout.rows(), features[0].cols, type_), layout);
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
