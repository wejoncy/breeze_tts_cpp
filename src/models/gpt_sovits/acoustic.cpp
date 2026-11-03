#include "models/gpt_sovits/acoustic.h"
#include "ops/attention.h"
namespace tts::gpt_sovits {
Acoustic::Acoustic(Weights &w, const Json &cfg)
    : phones_(w.get("acoustic.enc_p.text_embedding.weight")),
      ssl_encoder_(w, "acoustic.enc_p.encoder_ssl", cfg.at("n_layers").get<int>() / 2,
                   cfg.at("n_heads")),
      text_encoder_(w, "acoustic.enc_p.encoder_text", cfg.at("n_layers"), cfg.at("n_heads")),
      output_encoder_(w, "acoustic.enc_p.encoder2", cfg.at("n_layers").get<int>() / 2,
                      cfg.at("n_heads")),
      content_(w, "acoustic.enc_p.mrte.c_pre"), text_(w, "acoustic.enc_p.mrte.text_pre"),
      query_(w, "acoustic.enc_p.mrte.cross_attention.conv_q"),
      key_value_(
          w.fused("acoustic.mrte.kv.weight", {"acoustic.enc_p.mrte.cross_attention.conv_k.weight",
                                              "acoustic.enc_p.mrte.cross_attention.conv_v.weight"}),
          w.fused("acoustic.mrte.kv.bias", {"acoustic.enc_p.mrte.cross_attention.conv_k.bias",
                                            "acoustic.enc_p.mrte.cross_attention.conv_v.bias"})),
      attention_out_(w, "acoustic.enc_p.mrte.cross_attention.conv_o"),
      content_out_(w, "acoustic.enc_p.mrte.c_post"), bridge_(w, "acoustic.bridge.0"),
      waveform_features_(w, "acoustic.wns1", cfg.at("wn_layers"), cfg.at("wn_dilation_rate")),
      heads_(cfg.at("mrte_heads")), rate_(cfg.at("semantic_repeat"), cfg.at("feature_repeat")),
      type_(phones_.type) {
    for (const char *projection : {"conv_k", "conv_v"}) {
        auto name = std::string("acoustic.enc_p.mrte.cross_attention.") + projection;
        w.validate(name + ".weight", {query_.output_channels(), text_.output_channels(), 1}, type_);
        w.validate(name + ".bias", {query_.output_channels()}, type_);
    }
    if (heads_ < 1 || content_.output_channels() % heads_ ||
        cfg.at("gin_channels").get<int>() != content_.output_channels())
        throw std::invalid_argument("invalid acoustic rates/attention geometry");
    auto codebook = w.get("acoustic.quantizer.vq.layers.0._codebook.embed");
    Linear project(w, "acoustic.enc_p.ssl_proj");
    projected_codebook_ =
        Tensor({codebook.rows, project.output_channels()}, type_, &w.backend().memory);
    w.after_loading([project, codebook, output = projected_codebook_.view()](Backend &gpu) {
        project.forward(codebook, output);
    });
}
namespace {
int acoustic_rows(int batch, int time) {
    if (batch < 1 || time < 1 || int64_t(batch) * time > INT32_MAX)
        throw std::invalid_argument("invalid acoustic capacity");
    return batch * time;
}
Acoustic::Capacity checked_acoustic_capacity(Acoustic::Capacity c, int repeat) {
    if (c.sequences < 1 || c.codes < 1 || c.phones < 1 || c.frames < 1 || c.frames % repeat)
        throw std::invalid_argument("invalid acoustic capacity");
    return c;
}
}
Acoustic::Workspace::Workspace(const Acoustic &a, int batch, int codes, int phones,
                               BufferAllocator *allocator, int frames)
    : Workspace(a, Capacity{batch, acoustic_rows(batch, codes), acoustic_rows(batch, phones),
                            acoustic_rows(batch, frames ? frames : a.feature_frames(codes))}, allocator) {}
Acoustic::Workspace::Workspace(const Acoustic &a, Capacity cap, BufferAllocator *allocator)
    : ssl_encoder(a.ssl_encoder_, 1, a.hidden_frames(checked_acoustic_capacity(cap, a.rate_.feature_repeat).codes), allocator),
      text_encoder(a.text_encoder_, 1, cap.phones, allocator),
      output_encoder(a.output_encoder_, 1, a.hidden_frames(cap.codes), allocator),
      waveform_features(a.waveform_features_, 1, cap.frames, allocator, cap.sequences),
      max_batch(cap.sequences), max_codes(cap.codes), max_phones(cap.phones), max_frames(cap.frames) {
    if (cap.sequences < 1 || cap.codes < 1 || cap.phones < 1 || cap.frames < 1 ||
        cap.frames % a.rate_.feature_repeat)
        throw std::invalid_argument("invalid acoustic workspace");
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
    auto &gpu = backend_for(condition, {phones_});
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
    ops::embedding(gpu.stream, phones_, phones, text, f);
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
    auto &gpu = backend_for(features, {style, text_condition});
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
    ops::embedding(gpu.stream, projected_codebook_.view(), codes, ssl, f);
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
    dense_attention(gpu.stream, query, text_condition.columns(0, query.cols),
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
    ops::pointwise(gpu.stream, update, content, e);
    content_out_.forward(content, ssl);
    output_encoder_.forward(w.output_encoder, ssl, hidden_batch);
    int resized_time = input.resized.max_length, resized_rows = input.resized.rows;
    if (input.resample) {
        auto resized = w.content.view(resized_rows, ssl.cols, type_);
        ops::linear_resample(gpu.stream, ssl, resized, batch, time, resized_time,
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
    ops::pointwise(gpu.stream, update, features, e);
    waveform_features_.forward(w.waveform_features, features, style, features, feature_batch,
                               input.valid_frames, input.conditions, input.feature_outputs);
}
} // namespace tts::gpt_sovits
