#include "audio_kernels.h"
#include "codec.h"
#include <algorithm>
#include <iterator>

namespace tts {
Conv::Conv(Weights &weights, const std::string &name, int d, int s, bool trans, bool dw)
    : dilation(d), stride(s), transpose(trans), depthwise(dw) {
    const auto &e = weights.entry(name + ".weight");
    if (e.type != Type::F32 || e.shape.size() != 3)
        throw std::runtime_error("codec convolution requires F32");
    kernel = e.shape[2];
    in = trans ? e.shape[0] : e.shape[1];
    out = trans ? e.shape[1] : e.shape[0];
    if (dw)
        in = out;
    bias = weights.optional(name + ".bias");
    if (dw) {
        weight = weights.get(name + ".weight");
        return;
    }
    // Project each incoming sample through all taps at once, then accumulate
    // into the causal output ring. No input packing/zero insertion is needed.
    std::vector<float> reordered(size_t(out) * kernel * in);
    const float *src = static_cast<const float *>(e.data);
    for (int o = 0; o < out; o++)
        for (int k = 0; k < kernel; k++)
            for (int i = 0; i < in; i++) {
                size_t dest = (size_t(k) * out + o) * in + i;
                reordered[dest] = src[trans ? (size_t(i) * out + o) * kernel + k
                                            : (size_t(o) * in + i) * kernel + k];
            }
    weight = weights.custom(name + ".tap_major", reordered.data(), kernel * out, in, Type::F32);
}
void ConvState::run(Backend &gpu, const Conv &c, TensorView x, TensorView y, const int *frame) {
    if (c.depthwise) {
        auto h = history.view(batch * c.history(), c.in, Type::F32);
        tts::depthwise(gpu.stream, x, h, c.weight, c.bias, y, batch, time, c.kernel, frame);
    } else {
        auto p = projected.view(batch * time, c.kernel * c.out, Type::F32);
        gpu.linear(x, c.weight, p);
        auto pending = history.view(int(history.size() / 4 / c.out), c.out, Type::F32);
        tts::conv_accumulate(gpu.stream, p, pending, c.bias, y, batch, time, c.kernel, c.dilation, c.stride,
                            c.transpose, frame);
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
Codec::Codec(Weights &w, CodecConfig cfg)
    : pre_conv_(w, "codec.decoder.pre_conv.conv"), config(std::move(cfg)),
      transformer(w, "codec.decoder.pre_transformer", config.transformer) {
    std::string root = "codec.decoder.";
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
        CodecOp o{};
        o.kind = CodecOp::Convolution;
        o.conv = std::make_unique<Conv>(w, root + name + ".conv", dilation, stride, trans, dw);
        ops_.push_back(std::move(o));
    };
    auto op = [&](CodecOp::Kind kind, std::string name = "", int save = 0) {
        CodecOp o{};
        o.kind = kind;
        o.save = save;
        if (kind == CodecOp::Snake) {
            o.w = w.get(root + name + ".alpha");
            o.b = w.get(root + name + ".beta");
        } else if (kind == CodecOp::Linear || kind == CodecOp::LayerNorm) {
            o.w = w.get(root + name + ".weight");
            o.b = w.get(root + name + ".bias");
        } else if (kind == CodecOp::ScaleAdd)
            o.w = w.get(root + name + ".gamma");
        ops_.push_back(std::move(o));
    };
    for (size_t i = 0; i < config.upsampling_ratios.size(); i++) {
        std::string p = "upsample." + std::to_string(i);
        cv(p + ".0", 1, config.upsampling_ratios[i], true);
        p += ".1";
        op(CodecOp::Save, "", 0);
        cv(p + ".dwconv", 1, 1, false, true);
        op(CodecOp::LayerNorm, p + ".norm");
        op(CodecOp::Linear, p + ".pwconv1");
        op(CodecOp::Gelu);
        op(CodecOp::Linear, p + ".pwconv2");
        op(CodecOp::ScaleAdd, p, 0);
    }
    cv("decoder.0");
    if (ops_.back().conv->out != config.decoder_dim)
        throw std::invalid_argument("codec decoder_dim does not match its first convolution");
    for (size_t i = 0; i < config.upsample_rates.size(); i++) {
        std::string p = "decoder." + std::to_string(i + 1) + ".block.";
        op(CodecOp::Snake, p + "0");
        cv(p + "1", 1, config.upsample_rates[i], true);
        // Qwen3 DAC residual-unit definition, fixed by modeling_qwen3_tts_tokenizer_v2.py;
        // dilation is not stored in weight shapes or the checkpoint config.
        constexpr int dilations[] = {1, 3, 9};
        for (size_t j = 0; j < std::size(dilations); j++) {
            std::string r = p + std::to_string(j + 2);
            op(CodecOp::Save, "", 1);
            op(CodecOp::Snake, r + ".act1");
            cv(r + ".conv1", dilations[j]);
            op(CodecOp::Snake, r + ".act2");
            cv(r + ".conv2");
            op(CodecOp::Add, "", 1);
        }
    }
    op(CodecOp::Snake, "decoder." + std::to_string(config.upsample_rates.size() + 1));
    cv("decoder." + std::to_string(config.upsample_rates.size() + 2));
    if (pre_conv_.in != config.codebook_dim || pre_conv_.out != config.latent_dim ||
        first_proj_.rows != config.codebook_dim || rest_proj_.rows != config.codebook_dim ||
        first_proj_.cols != first_embedding_.cols || rest_proj_.cols != rest_embedding_.cols ||
        input_w_.rows != config.transformer.hidden || input_w_.cols != config.latent_dim ||
        output_w_.rows != config.latent_dim || output_w_.cols != config.transformer.hidden)
        throw std::invalid_argument("codec projection weight/configuration mismatch");
}
CodecState::CodecState(const Codec &c, int b, BufferAllocator *arena)
    : transformer(c.transformer.config, b, c.transformer.config.window, b, arena, {}, true),
      rvq1(size_t(b) * c.first_embedding_.cols * sizeof(float), arena), rvq2(rvq1.size(), arena),
      latent(size_t(b) * c.config.latent_dim * sizeof(float), arena),
      hidden(size_t(b) * c.config.transformer.hidden * sizeof(float), arena),
      position(size_t(b) * 4, arena),
      pcm(size_t(b) * c.config.samples_per_frame * sizeof(int16_t), arena), batch(b) {
    int time = 1, channels = c.config.latent_dim;
    max_elements = b * std::max(c.config.latent_dim, c.config.codebook_dim);
    conv.push_back(std::make_unique<ConvState>(c.pre_conv_, b, 1, arena));
    for (const auto &op : c.ops_) {
        if (op.kind == CodecOp::Convolution) {
            if (op.conv->in != channels ||
                (op.conv->transpose && time > INT32_MAX / op.conv->stride))
                throw std::invalid_argument("codec convolution chain shape");
            conv.push_back(std::make_unique<ConvState>(*op.conv, b, time, arena));
            if (op.conv->transpose)
                time *= op.conv->stride;
            channels = op.conv->out;
        }
        if (op.kind == CodecOp::Linear)
            channels = op.w.rows;
        int64_t elements = int64_t(b) * time * channels;
        if (elements > INT32_MAX)
            throw std::invalid_argument("codec workspace exceeds kernel indexing capacity");
        max_elements = std::max(max_elements, int(elements));
    }
    if (time != c.config.samples_per_frame || channels != 1)
        throw std::invalid_argument("codec output does not match configured frame size");
    a = Buffer(size_t(max_elements) * 4, arena);
    this->b = Buffer(a.size(), arena);
    for (auto &save : saves)
        save = Buffer(a.size(), arena);
}
void CodecState::reset(Backend &gpu) {
    gpu.zero(position);
    for (auto &c : conv)
        gpu.zero(c->history);
}
TensorView Codec::forward(Backend &gpu, CodecState &s, const int *codes) const {
    int batch = s.batch;
    auto a = s.a.view(batch, config.codebook_dim, Type::F32),
         b = s.b.view(batch, config.codebook_dim, Type::F32);
    auto r1 = s.rvq1.view(batch, first_embedding_.cols, Type::F32),
         r2 = s.rvq2.view(batch, rest_embedding_.cols, Type::F32);
    tts::rvq(gpu.stream, first_embedding_, codes, r1, batch, 0, config.semantic_codebooks,
             config.codebooks);
    tts::rvq(gpu.stream, rest_embedding_, codes, r2, batch, config.semantic_codebooks,
             config.codebooks - config.semantic_codebooks, config.codebooks);
    gpu.linear(r1, first_proj_, a);
    gpu.linear(r2, rest_proj_, b);
    gpu.add(a, b, a);
    auto latent = s.latent.view(batch, config.latent_dim, Type::F32),
         hidden = s.hidden.view(batch, config.transformer.hidden, Type::F32);
    s.conv[0]->run(gpu, pre_conv_, a, latent, s.position.ptr<int>());
    gpu.linear(latent, input_w_, hidden);
    gpu.bias(hidden, input_b_);
    transformer.forward(gpu, s.transformer, hidden, s.position.ptr<int>(), batch, 1);
    a = s.a.view(batch, config.latent_dim, Type::F32);
    gpu.linear(hidden, output_w_, a);
    gpu.bias(a, output_b_);
    int ci = 1, time = 1, channels = config.latent_dim, current = 0;
    TensorView saved[2];
    int saved_index[2] = {-1, -1};
    const Buffer *buffers[] = {&s.a, &s.b, &s.saves[0], &s.saves[1]};
    for (const auto &op : ops_) {
        TensorView x = buffers[current]->view(batch * time, channels, Type::F32);
        int newtime = time, newchannels = channels;
        if (op.kind == CodecOp::Save) {
            saved[op.save] = x;
            saved_index[op.save] = current;
            continue;
        }
        if (op.kind == CodecOp::Convolution) {
            newtime = time * (op.conv->transpose ? op.conv->stride : 1);
            newchannels = op.conv->out;
        }
        if (op.kind == CodecOp::Linear)
            newchannels = op.w.rows;
        int next = 0;
        while (next == current || next == saved_index[0] || next == saved_index[1])
            ++next;
        if (next >= 4)
            throw std::runtime_error("codec buffer liveness invariant");
        TensorView y = buffers[next]->view(batch * newtime, newchannels, Type::F32);
        switch (op.kind) {
        case CodecOp::Convolution:
            s.conv[ci++]->run(gpu, *op.conv, x, y, s.position.ptr<int>());
            break;
        case CodecOp::Snake:
            tts::snake(gpu.stream, x, op.w, op.b, y);
            break;
        case CodecOp::Add:
            gpu.add(x, saved[op.save], y);
            break;
        case CodecOp::ScaleAdd:
            gpu.add(saved[op.save], x, y, op.w);
            break;
        case CodecOp::LayerNorm:
            gpu.norm(x, op.w, y, 1e-6f, false, op.b);
            break;
        case CodecOp::Linear:
            gpu.linear(x, op.w, y);
            gpu.bias(y, op.b);
            break;
        case CodecOp::Gelu:
            gpu.activation(x, y, 2);
            break;
        default:
            break;
        }
        if (op.kind == CodecOp::Add || op.kind == CodecOp::ScaleAdd)
            saved_index[op.save] = -1;
        current = next;
        time = newtime;
        channels = newchannels;
    }
    auto result = buffers[current]->view(batch * config.samples_per_frame, 1, Type::F32);
    tts::pcm(gpu.stream, result, s.pcm.ptr<int16_t>());
    gpu.increment(s.position.ptr<int>(), batch);
    return result;
}
} // namespace tts
