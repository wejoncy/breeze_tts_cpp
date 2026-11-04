#include "ops/basic.h"
#include "ops/audio.h"
#include "models/mimi/decoder.h"
#include "layers/convolution.h"
#include <algorithm>
#include <iterator>

namespace tts {
Codec::Conv::Conv(Weights &weights, const std::string &name, int dilation, int stride,
           bool transpose, bool depthwise)
    : dilation(dilation), stride(stride), transpose(transpose), depthwise(depthwise) {
    const auto &entry = weights.entry(name + ".weight");
    if (entry.type != Type::F32 || entry.shape.size() != 3 || entry.shape[0] < 1 || entry.shape[1] < 1 ||
        entry.shape[2] < 1 || dilation < 1 || stride < 1 ||
        (depthwise && (transpose || entry.shape[1] != 1)) || (transpose && entry.shape[2] < stride))
        throw std::runtime_error("codec convolution requires F32");
    kernel = entry.shape[2];
    in = transpose ? entry.shape[0] : entry.shape[1];
    out = transpose ? entry.shape[1] : entry.shape[0];
    if (depthwise)
        in = out;
    bias = weights.get(name + ".bias");
    if (bias.type != Type::F32 || bias.numel() != size_t(out))
        throw std::invalid_argument("codec convolution bias mismatch: " + name);
    if (depthwise) {
        weight = weights.get(name + ".weight");
        return;
    }
    // Project each incoming sample through all taps at once, then accumulate
    // into the causal output ring. No input packing/zero insertion is needed.
    weight = load_convolution_weight(weights, name, transpose);
}
void Codec::Conv::forward(TensorView input, TensorView output, TensorView projection,
                           int batch, int time, const ConvHistory *histories) const {
    if (depthwise) {
        tts::depthwise(input, {}, weight, bias, output, batch, time, kernel, nullptr, histories);
    } else {
        ops::matmul(input, weight, projection);
        tts::conv_accumulate(projection, {}, bias, output, batch, time, kernel, dilation,
                             stride, transpose, nullptr, histories);
    }
}
static TensorView codebook(Weights &weights, const std::string &prefix, int count, int entries,
                           int dimension) {
    std::vector<float> table(size_t(count) * entries * dimension);
    for (int k = 0; k < count; k++) {
        std::string p = prefix + ".vq.layers." + std::to_string(k) + "._codebook.";
        auto &e = weights.entry(p + "embedding_sum");
        auto &u = weights.entry(p + "cluster_usage");
        if (e.shape != std::vector<int>({entries, dimension}) || e.type != Type::F32 ||
            u.type != Type::F32 || u.shape != std::vector<int>{entries})
            throw std::runtime_error("unsupported codec codebook");
        const auto *sum = static_cast<const float *>(e.data);
        const auto *usage = static_cast<const float *>(u.data);
        for (int i = 0; i < entries; i++)
            for (int d = 0; d < dimension; d++)
                table[(size_t(k) * entries + i) * dimension + d] =
                    sum[size_t(i) * dimension + d] / std::max(usage[i], 1e-5f);
    }
    return weights.custom(prefix + ".normalized", table.data(), count * entries, dimension,
                          Type::F32);
}
Codec::Codec(Weights &w, const std::string &prefix, CodecConfig cfg)
    : pre_conv_(w, prefix + ".pre_conv.conv"), config(std::move(cfg)),
      transformer(w, prefix + ".pre_transformer", config.transformer) {
    std::string root = prefix + ".";
    first_embedding_ = codebook(w, root + "quantizer.rvq_first", config.semantic_codebooks,
                                config.codebook_size, config.codebook_dim / 2);
    rest_embedding_ =
        codebook(w, root + "quantizer.rvq_rest", config.codebooks - config.semantic_codebooks,
                 config.codebook_size, config.codebook_dim / 2);
    first_proj_ = w.get(root + "quantizer.rvq_first.output_proj.weight");
    rest_proj_ = w.get(root + "quantizer.rvq_rest.output_proj.weight");
    input_w_ = w.get(root + "pre_transformer.input_proj.weight");
    input_b_ = w.get(root + "pre_transformer.input_proj.bias");
    output_w_ = w.get(root + "pre_transformer.output_proj.weight");
    output_b_ = w.get(root + "pre_transformer.output_proj.bias");
    auto cv = [&](std::string name, int dilation = 1, int stride = 1, bool trans = false,
                  bool dw = false) {
        Operation o{};
        o.kind = Operation::Convolution;
        o.conv = std::make_unique<Conv>(w, root + name + ".conv", dilation, stride, trans, dw);
        ops_.push_back(std::move(o));
    };
    auto op = [&](Operation::Kind kind, std::string name = "", int save = 0) {
        Operation o{};
        o.kind = kind;
        o.save = save;
        if (kind == Operation::Snake) {
            o.weight = w.get(root + name + ".alpha");
            o.bias = w.get(root + name + ".beta");
        } else if (kind == Operation::Linear || kind == Operation::LayerNorm) {
            o.weight = w.get(root + name + ".weight");
            o.bias = w.get(root + name + ".bias");
        } else if (kind == Operation::ScaleAdd)
            o.weight = w.get(root + name + ".gamma");
        ops_.push_back(std::move(o));
    };
    for (size_t i = 0; i < config.upsampling_ratios.size(); i++) {
        std::string p = "upsample." + std::to_string(i);
        cv(p + ".0", 1, config.upsampling_ratios[i], true);
        p += ".1";
        op(Operation::Save, "", 0);
        cv(p + ".dwconv", 1, 1, false, true);
        op(Operation::LayerNorm, p + ".norm");
        op(Operation::Linear, p + ".pwconv1");
        op(Operation::Gelu);
        op(Operation::Linear, p + ".pwconv2");
        op(Operation::ScaleAdd, p, 0);
    }
    cv("decoder.0");
    if (ops_.back().conv->out != config.decoder_dim)
        throw std::invalid_argument("codec decoder_dim does not match its first convolution");
    for (size_t i = 0; i < config.upsample_rates.size(); i++) {
        std::string p = "decoder." + std::to_string(i + 1) + ".block.";
        op(Operation::Snake, p + "0");
        cv(p + "1", 1, config.upsample_rates[i], true);
        // Qwen3 DAC residual-unit definition, fixed by modeling_qwen3_tts_tokenizer_v2.py;
        // dilation is not stored in weight shapes or the checkpoint config.
        constexpr int dilations[] = {1, 3, 9};
        for (size_t j = 0; j < std::size(dilations); j++) {
            std::string r = p + std::to_string(j + 2);
            op(Operation::Save, "", 1);
            op(Operation::Snake, r + ".act1");
            cv(r + ".conv1", dilations[j]);
            op(Operation::Snake, r + ".act2");
            cv(r + ".conv2");
            op(Operation::Add, "", 1);
        }
    }
    op(Operation::Snake, "decoder." + std::to_string(config.upsample_rates.size() + 1));
    cv("decoder." + std::to_string(config.upsample_rates.size() + 2));
    int channels = config.latent_dim, saved[2]{};
    for (const auto &o : ops_) {
        if ((o.weight.data && o.weight.type != Type::F32) || (o.bias.data && o.bias.type != Type::F32))
            throw std::invalid_argument("codec operator requires F32 parameters");
        if (o.kind == Operation::Convolution) {
            if (o.conv->in != channels)
                throw std::invalid_argument("codec convolution chain shape");
            channels = o.conv->out;
        } else if (o.kind == Operation::Linear) {
            if (o.weight.cols != channels || o.bias.numel() != size_t(o.weight.rows))
                throw std::invalid_argument("codec linear weight/bias shape");
            channels = o.weight.rows;
        } else {
            if ((o.weight.data && o.weight.numel() != size_t(channels)) ||
                (o.bias.data && o.bias.numel() != size_t(channels)))
                throw std::invalid_argument("codec channel parameter shape");
            if (o.kind == Operation::Save) saved[o.save] = channels;
            if (o.kind == Operation::Add || o.kind == Operation::ScaleAdd) {
                if (saved[o.save] != channels)
                    throw std::invalid_argument("codec residual channel shape");
                saved[o.save] = 0;
            }
        }
    }
    for (auto value : {first_proj_, rest_proj_, input_w_, input_b_, output_w_, output_b_})
        if (value.type != Type::F32)
            throw std::invalid_argument("codec projection requires F32 parameters");
    if (pre_conv_.in != config.codebook_dim || pre_conv_.out != config.latent_dim ||
        first_proj_.rows != config.codebook_dim || rest_proj_.rows != config.codebook_dim ||
        first_proj_.cols != first_embedding_.cols || rest_proj_.cols != rest_embedding_.cols ||
        input_w_.rows != config.transformer.hidden || input_w_.cols != config.latent_dim ||
        input_b_.numel() != size_t(config.transformer.hidden) ||
        output_w_.rows != config.latent_dim || output_w_.cols != config.transformer.hidden ||
        output_b_.numel() != size_t(config.latent_dim) || channels != 1)
        throw std::invalid_argument("codec projection weight/configuration mismatch");
}
SequenceStateSpec Codec::history_spec() const {
    SequenceStateSpec spec;
    spec.layer_bytes = size_t(transformer.config.window) * transformer.config.kvheads *
                       transformer.config.dim * bytes(transformer.config.type);
    spec.kv_bytes = size_t(transformer.config.layers) * spec.layer_bytes;
    spec.history_bytes.push_back(pre_conv_.history_bytes(1, 1));
    int time = 1;
    for (const auto &op : ops_)
        if (op.kind == Operation::Convolution) {
            spec.history_bytes.push_back(op.conv->history_bytes(1, time));
            if (op.conv->transpose)
                time *= op.conv->stride;
        }
    return spec;
}
Codec::Workspace::Workspace(const Codec &model, int batch)
    : rvq_first(size_t(batch) * model.first_embedding_.cols * sizeof(float)),
      rvq_rest(rvq_first.size()), latent(size_t(batch) * model.config.latent_dim * sizeof(float)),
      hidden(size_t(batch) * model.config.transformer.hidden * sizeof(float)),
      pcm(size_t(batch) * model.config.samples_per_frame * sizeof(int16_t)) {
    int time = 1, channels = model.config.latent_dim;
    int64_t max_elements = int64_t(batch) * std::max(channels, model.config.codebook_dim);
    size_t projection_elements = size_t(batch) * model.pre_conv_.kernel * model.pre_conv_.out;
    for (const auto &operation : model.ops_) {
        if (operation.kind == Operation::Convolution) {
            const auto &convolution = *operation.conv;
            if (!convolution.depthwise)
                projection_elements = std::max(projection_elements,
                    size_t(batch) * time * convolution.kernel * convolution.out);
            if (convolution.transpose) {
                if (time > INT32_MAX / convolution.stride)
                    throw std::invalid_argument("codec convolution time overflow");
                time *= convolution.stride;
            }
            channels = convolution.out;
        }
        if (operation.kind == Operation::Linear)
            channels = operation.weight.rows;
        max_elements = std::max(max_elements, int64_t(batch) * time * channels);
    }
    if (max_elements > INT32_MAX || projection_elements > INT32_MAX)
        throw std::invalid_argument("codec workspace exceeds kernel indexing capacity");
    if (time != model.config.samples_per_frame || channels != 1)
        throw std::invalid_argument("codec output does not match configured frame size");
    for (auto &activation : activations)
        activation = Buffer(size_t(max_elements) * sizeof(float));
    convolution = Buffer(projection_elements * sizeof(float));
}
void Codec::forward(TensorInputs inputs, TensorOutputs outputs, ForwardContext &context) const {
    Workspace workspace(*this, context.max_batch);
    const int batch = context.batch.batch();
    const auto *code_rows = inputs["code_rows"].ptr<const int *>();
    const auto *histories = inputs["histories"].ptr<ConvHistory>();
    const int history_stride = int(context.parameters["history_stride"]);
    const auto *cache = context.batch.cache.layout.sequences;
    const bool execute = context.mode == ForwardContext::Execute;
    if (execute && (!code_rows || !histories || !cache || history_stride < batch))
        throw std::invalid_argument("codec requires runner-owned history and cache rows");
    auto convolve = [&](const Conv &convolution, TensorView input, TensorView output,
                        int time, int history_index) {
        if (!execute)
            return;
        auto projection = convolution.depthwise ? TensorView{} : workspace.convolution.view(
            batch * time, convolution.kernel * convolution.out, Type::F32);
        convolution.forward(input, output, projection, batch, time, histories + history_index * history_stride);
    };
    auto first = workspace.activations[0].view(batch, config.codebook_dim, Type::F32);
    auto second = workspace.activations[1].view(batch, config.codebook_dim, Type::F32);
    auto first_codes = workspace.rvq_first.view(batch, first_embedding_.cols, Type::F32);
    auto rest_codes = workspace.rvq_rest.view(batch, rest_embedding_.cols, Type::F32);
    auto latent = workspace.latent.view(batch, config.latent_dim, Type::F32);
    auto hidden = workspace.hidden.view(batch, config.transformer.hidden, Type::F32);
    if (execute) {
        tts::rvq(first_embedding_, nullptr, first_codes, batch, 0, config.semantic_codebooks,
                  config.codebooks, code_rows);
        tts::rvq(rest_embedding_, nullptr, rest_codes, batch, config.semantic_codebooks,
                  config.codebooks - config.semantic_codebooks, config.codebooks, code_rows);
        ops::matmul(first_codes, first_proj_, first);
        ops::matmul(rest_codes, rest_proj_, second);
        ops::add(first, second, first);
    }
    convolve(pre_conv_, first, latent, 1, 0);
    if (execute) {
        ops::matmul(latent, input_w_, hidden);
        ops::bias(hidden, input_b_);
    }
    auto decoder_context = context;
    decoder_context.batch = ForwardBatch{batch, 1};
    decoder_context.batch.cache.layout.sequences = cache;
    decoder_context.batch.cache.layout.ring = true;
    decoder_context.batch.forward_mode = ForwardMode::Decode;
    decoder_context.max_time = config.transformer.window;
    transformer.forward(hidden, decoder_context);
    first = workspace.activations[0].view(batch, config.latent_dim, Type::F32);
    if (execute) {
        ops::matmul(hidden, output_w_, first);
        ops::bias(first, output_b_);
    }
    int convolution_index = 1, time = 1, channels = config.latent_dim, current = 0;
    TensorView saved[2];
    int saved_indices[2] = {-1, -1};
    for (const auto &operation : ops_) {
        auto input = workspace.activations[current].view(batch * time, channels, Type::F32);
        int next_time = time, next_channels = channels;
        if (operation.kind == Operation::Save) {
            saved[operation.save] = input;
            saved_indices[operation.save] = current;
            continue;
        }
        if (operation.kind == Operation::Convolution) {
            next_time = time * (operation.conv->transpose ? operation.conv->stride : 1);
            next_channels = operation.conv->out;
        }
        if (operation.kind == Operation::Linear)
            next_channels = operation.weight.rows;
        int next = 0;
        while (next == current || next == saved_indices[0] || next == saved_indices[1])
            ++next;
        if (next >= int(workspace.activations.size()))
            throw std::runtime_error("codec buffer liveness invariant");
        auto output = workspace.activations[next].view(batch * next_time, next_channels, Type::F32);
        if (execute) {
            switch (operation.kind) {
            case Operation::Convolution:
                convolve(*operation.conv, input, output, time, convolution_index++);
                break;
            case Operation::Snake:
                tts::snake(input, operation.weight, operation.bias, output);
                break;
            case Operation::Add:
                ops::add(input, saved[operation.save], output);
                break;
            case Operation::ScaleAdd:
                ops::add(saved[operation.save], input, output, operation.weight);
                break;
            case Operation::LayerNorm:
                ops::norm(input, operation.weight, output, 1e-6f, false, operation.bias);
                break;
            case Operation::Linear:
                ops::matmul(input, operation.weight, output);
                ops::bias(output, operation.bias);
                break;
            case Operation::Gelu:
                ops::activation(input, output, 2);
                break;
            default:
                break;
            }
        }
        if (operation.kind == Operation::Add || operation.kind == Operation::ScaleAdd)
            saved_indices[operation.save] = -1;
        current = next;
        time = next_time;
        channels = next_channels;
    }
    auto waveform = workspace.activations[current].view(batch * config.samples_per_frame, 1, Type::F32);
    if (execute) {
        tts::pcm(waveform, workspace.pcm.ptr<int16_t>());
        ops::advance_positions(workspace.activations[0].view(1, 1, Type::F32), nullptr, batch, 1, cache);
    }
    outputs["waveform"] = waveform;
    outputs["pcm"] = workspace.pcm.view(batch, config.samples_per_frame, Type::I16);
}
} // namespace tts
