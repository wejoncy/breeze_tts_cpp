#include "models/breeze/audio_encoder.h"
#include <filesystem>
#include "kernels/audio/encoder_kernels.h"
#include <cmath>
#include <algorithm>
#include <fstream>
namespace tts {
EncoderConv::EncoderConv(Weights &w, const std::string &p, int s, bool rep, int d)
    : stride(s), dilation(d), replicate(rep) {
    const auto &e = w.entry(p + ".conv.weight");
    if (stride < 1 || dilation < 1 || (stride > 1 && dilation > 1) || e.type != Type::F32 ||
        e.shape.size() != 3 || e.shape[2] % stride)
        throw std::runtime_error("unsupported encoder convolution");
    out = e.shape[0];
    in = e.shape[1];
    kernel = e.shape[2];
    if (kernel > INT32_MAX / out || int64_t(kernel - 1) * dilation >= INT32_MAX)
        throw std::invalid_argument("encoder convolution indexing overflow");
    bias = w.optional(p + ".conv.bias");
    if (bias.data && (bias.type != Type::F32 || bias.numel() != size_t(out)))
        throw std::invalid_argument("encoder convolution bias shape: " + p);
    std::vector<float> packed(e.size / 4);
    const float *src = static_cast<const float *>(e.data);
    int taps = kernel / stride;
    for (int phase = 0; phase < stride; phase++)
        for (int tap = 0; tap < taps; tap++)
            for (int o = 0; o < out; o++)
                for (int c = 0; c < in; c++)
                    packed[((size_t(phase) * taps + tap) * out + o) * in + c] =
                        src[(size_t(o) * in + c) * kernel + tap * stride + phase];
    weight = w.custom(p + ".phase_major", packed.data(), kernel * out, in, Type::F32);
}
void EncoderConv::run(Backend &g, TensorView x, TensorView scratch, TensorView y, bool elu,
                      TensorView residual) const {
    int n = rows(x.rows), taps = kernel / stride;
    if (x.cols != in || y.rows != n || y.cols != out)
        throw std::runtime_error("encoder convolution shape");
    auto p = scratch.reshape({int(scratch.numel() / (kernel * out)), kernel * out}).slice(0, n);
    for (int phase = 0; phase < stride && phase < x.rows; phase++) {
        int count = (x.rows - 1 - phase) / stride + 1;
        g.linear(x.every_rows(phase, stride, count), weight.slice(phase * taps * out, taps * out),
                 p.columns(phase * taps * out, taps * out).slice(0, count), true);
    }
    encoder_conv_finish(g.stream, x, p, weight, bias, y, kernel, stride, replicate, elu, residual,
                        dilation);
}
AudioEncoder::Residual::Residual(Weights &w, int i, int dilation)
    : first(w, "codec.encoder.encoder.layers." + std::to_string(i) + ".block.1", 1, false,
            dilation),
      second(w, "codec.encoder.encoder.layers." + std::to_string(i) + ".block.3") {}
AudioEncoder::Stage::Stage(Weights &w, int i, int ratio, const EncoderConfig &c)
    : down(w, "codec.encoder.encoder.layers." + std::to_string(i + c.residual_layers + 1), ratio) {
    int dilation = 1;
    for (int j = 0; j < c.residual_layers; ++j) {
        blocks.emplace_back(w, i + j, dilation);
        const auto &b = blocks.back();
        if (b.first.in != down.in || b.first.out != down.in / c.compress ||
            b.second.in != b.first.out || b.second.out != down.in ||
            b.first.kernel != c.residual_kernel || b.second.kernel != 1)
            throw std::invalid_argument("encoder residual weight/configuration mismatch");
        if (j + 1 < c.residual_layers) {
            if (dilation > INT32_MAX / c.dilation_growth)
                throw std::invalid_argument("encoder dilation overflow");
            dilation *= c.dilation_growth;
        }
    }
    if (down.kernel != 2 * ratio)
        throw std::invalid_argument("encoder downsampling weight/configuration mismatch");
}
AudioEncoder::AudioEncoder(Weights &w, EncoderConfig config)
    : config_(std::move(config)), first_(w, "codec.encoder.encoder.layers.0"),
      // Mimi's ModuleList: first conv, (residual blocks, ELU, downconv)*stages, ELU, last conv.
      last_(w, "codec.encoder.encoder.layers." +
                   std::to_string(2 + config_.ratios.size() * (config_.residual_layers + 2))),
      rope_storage_({config_.transformer.max_positions, config_.transformer.dim}, Type::F32,
                    &w.backend().memory) {
    const auto &c = config_.transformer;
    if (first_.in != 1 || first_.out != config_.filters || first_.kernel != config_.first_kernel ||
        last_.out != c.hidden || last_.kernel != config_.last_kernel)
        throw std::invalid_argument("encoder boundary convolution weight/configuration mismatch");
    int index = 1, channels = first_.out;
    for (int ratio : config_.ratios) {
        stages_.emplace_back(w, index, ratio, config_);
        if (stages_.back().down.in != channels || stages_.back().down.out != 2 * channels)
            throw std::invalid_argument("encoder stage channel mismatch");
        channels = stages_.back().down.out;
        index += config_.residual_layers + 2;
    }
    if (last_.in != channels)
        throw std::invalid_argument("encoder last convolution channel mismatch");
    if (config_.downsample_stride != 1) {
        downsample_ = std::make_unique<EncoderConv>(w, "codec.encoder.downsample",
                                                    config_.downsample_stride, true);
        if (downsample_->in != c.hidden || downsample_->out != c.hidden ||
            downsample_->kernel != 2 * config_.downsample_stride)
            throw std::invalid_argument("encoder resampling weight/configuration mismatch");
    }
    auto matrix = [&](const std::string &name, int rows, int cols, bool upload = true) {
        const auto &e = w.entry(name);
        if (e.type != Type::F32 || e.shape != std::vector<int>{rows, cols})
            throw std::invalid_argument("encoder matrix shape: " + name);
        return upload ? w.get(name) : TensorView{};
    };
    auto vector = [&](const std::string &name) {
        const auto &e = w.entry(name);
        if (e.type != Type::F32 || e.shape != std::vector<int>{c.hidden})
            throw std::invalid_argument("encoder vector shape: " + name);
        return w.get(name);
    };
    for (int i = 0; i < c.layers; i++) {
        std::string p = "codec.encoder.encoder_transformer.layers." + std::to_string(i);
        Layer l;
        l.pre = vector(p + ".input_layernorm.weight");
        l.pre_bias = vector(p + ".input_layernorm.bias");
        l.post = vector(p + ".post_attention_layernorm.weight");
        l.post_bias = vector(p + ".post_attention_layernorm.bias");
        matrix(p + ".self_attn.q_proj.weight", c.heads * c.dim, c.hidden, false);
        matrix(p + ".self_attn.k_proj.weight", c.kvheads * c.dim, c.hidden, false);
        matrix(p + ".self_attn.v_proj.weight", c.kvheads * c.dim, c.hidden, false);
        l.qkv = w.fused(p + ".qkv", {p + ".self_attn.q_proj.weight", p + ".self_attn.k_proj.weight",
                                     p + ".self_attn.v_proj.weight"});
        l.out = matrix(p + ".self_attn.o_proj.weight", c.hidden, c.heads * c.dim);
        l.fc1 = matrix(p + ".mlp.fc1.weight", c.intermediate, c.hidden);
        l.fc2 = matrix(p + ".mlp.fc2.weight", c.hidden, c.intermediate);
        l.attn_scale = vector(p + ".self_attn_layer_scale.scale");
        l.mlp_scale = vector(p + ".mlp_layer_scale.scale");
        layers_.push_back(l);
    }
    semantic_proj_ =
        w.get("codec.encoder.quantizer.semantic_residual_vector_quantizer.input_proj.weight");
    acoustic_proj_ =
        w.get("codec.encoder.quantizer.acoustic_residual_vector_quantizer.input_proj.weight");
    for (auto projection : {semantic_proj_, acoustic_proj_})
        if (projection.type != Type::F32 || projection.rows != config_.codebook_dim ||
            projection.cols != c.hidden)
            throw std::invalid_argument("encoder quantizer projection shape");
    int entries = config_.codebook_size, dimension = config_.codebook_dim;
    for (int i = 0; i < config_.codebooks; i++) {
        std::string p =
            "codec.encoder.quantizer." +
            std::string(i >= config_.semantic_codebooks ? "acoustic" : "semantic") +
            "_residual_vector_quantizer.layers." +
            std::to_string(i >= config_.semantic_codebooks ? i - config_.semantic_codebooks : i) +
            ".codebook.";
        const auto &e = w.entry(p + "embed_sum");
        const auto &usage = w.entry(p + "cluster_usage");
        if (e.shape != std::vector<int>({entries, dimension}) || e.type != Type::F32 ||
            usage.type != Type::F32 || usage.shape != std::vector<int>({entries}))
            throw std::runtime_error("encoder codebook shape");
        const auto *src = static_cast<const float *>(e.data);
        const auto *u = static_cast<const float *>(usage.data);
        std::vector<float> table(size_t(entries) * dimension), norm(entries);
        for (int k = 0; k < entries; k++)
            for (int d = 0; d < dimension; d++) {
                float v = src[size_t(k) * dimension + d] / std::max(u[k], 1e-5f);
                table[size_t(k) * dimension + d] = v;
                norm[k] += v * v;
            }
        tables_.push_back(w.custom(p + "normalized", table.data(), entries, dimension, Type::F32));
        norms_.push_back(w.custom(p + "norm", norm.data(), 1, entries, Type::F32));
    }
    rope_ = rope_storage_.view();
    w.backend().make_rope(rope_, c.dim, c.theta, 1, 0);
}
// Planning and allocation use the same shapes and alignment. No model-sized upper bounds.
struct AudioEncoder::Storage {
    Buffer input, a, b, c, projected, qkv, q, k, v, att, ff, residual, scores, codes;
    Storage(const AudioEncoder &model, int samples, BufferAllocator *arena) {
        const auto &config = model.config_;
        const auto &tr = config.transformer;
        if (samples <= 0 ||
            samples / config.stride + (samples % config.stride != 0) > tr.max_positions)
            throw std::invalid_argument("reference audio is empty or exceeds encoder positions");
        size_t elements = size_t(samples) * model.first_.out, scratch = 0;
        auto conv = [&](const EncoderConv &op, int rows) {
            int n = op.rows(rows);
            elements = std::max({elements, size_t(rows) * op.in, size_t(n) * op.out});
            scratch = std::max(scratch, size_t(n) * op.kernel * op.out);
            return n;
        };
        int rows = samples;
        for (const auto &stage : model.stages_) {
            for (const auto &block : stage.blocks) {
                conv(block.first, rows);
                conv(block.second, rows);
            }
            rows = conv(stage.down, rows);
        }
        conv(model.last_, rows);
        int frames = model.downsample_ ? conv(*model.downsample_, rows) : rows;
        if (elements > INT32_MAX || scratch > INT32_MAX)
            throw std::invalid_argument("encoder workspace exceeds kernel indexing capacity");
        input = Buffer(size_t(samples) * sizeof(float), arena);
        a = Buffer(elements * sizeof(float), arena);
        b = Buffer(a.size(), arena);
        c = Buffer(a.size(), arena);
        projected = Buffer(scratch * sizeof(float), arena);
        qkv = Buffer(size_t(rows) * (tr.heads + 2 * tr.kvheads) * tr.dim * sizeof(float), arena);
        q = Buffer(size_t(rows) * tr.heads * tr.dim * sizeof(float), arena);
        k = Buffer(size_t(rows) * tr.kvheads * tr.dim * sizeof(float), arena);
        v = Buffer(k.size(), arena);
        att = Buffer(q.size(), arena);
        ff = Buffer(size_t(rows) * tr.intermediate * sizeof(float), arena);
        residual = Buffer(size_t(frames) * config.codebook_dim * sizeof(float), arena);
        scores = Buffer(size_t(frames) * config.codebook_size * sizeof(float), arena);
        codes = Buffer(size_t(frames) * config.codebooks * sizeof(int), arena);
    }
};
size_t AudioEncoder::workspace_bytes(int samples) const {
    WorkspacePlan plan;
    Storage storage(*this, samples, &plan);
    return plan.size;
}
std::vector<int> AudioEncoder::encode(Backend &g, BufferAllocator &arena,
                                      const std::vector<float> &mono) const {
    Range range("tts.reference_encoder");
    if (mono.size() > INT32_MAX)
        throw std::invalid_argument("reference audio sample count exceeds tensor indexing");
    int samples = int(mono.size());
    const auto &cfg = config_.transformer;
    auto [input, a, b, c, projected, qkv, q, k, v, att, ff, residual, scores, codes] =
        Storage(*this, samples, &arena);
    auto workspace = [&](const EncoderConv &cv, int n) {
        return projected.view(cv.rows(n), cv.kernel * cv.out, Type::F32);
    };
    g.upload(input.data(), mono.data(), input.size());
    auto x = a.view(samples, first_.out, Type::F32);
    encoder_first_conv(g.stream, input.view(samples, first_.in, Type::F32), first_.weight,
                       first_.bias, x, first_.kernel);
    for (const auto &stage : stages_) {
        auto activated = b.view(x.rows, x.cols, Type::F32);
        g.activation(x, activated, 3);
        for (size_t i = 0; i < stage.blocks.size(); ++i) {
            const auto &block = stage.blocks[i];
            auto mid = c.view(x.rows, block.first.out, Type::F32);
            block.first.run(g, activated, workspace(block.first, x.rows), mid, true);
            // Keep the residual unactivated until all residual blocks have consumed it.
            bool last = i + 1 == stage.blocks.size();
            block.second.run(g, mid, workspace(block.second, mid.rows), last ? activated : x, last,
                             x);
            if (!last)
                g.activation(x, activated, 3);
        }
        auto next = a.view(stage.down.rows(x.rows), stage.down.out, Type::F32);
        stage.down.run(g, activated, workspace(stage.down, x.rows), next);
        x = next;
    }
    auto act = b.view(x.rows, x.cols, Type::F32);
    g.activation(x, act, 3);
    x = a.view(act.rows, last_.out, Type::F32);
    last_.run(g, act, workspace(last_, act.rows), x);
    auto dump = [&](TensorView value, const char *suffix) {
        if (const char *prefix = std::getenv("TTS_ENCODER_DUMP")) {
            std::vector<float> h(value.numel());
            g.download(h.data(), value.data, value.size());
            std::ofstream f(std::filesystem::u8path(std::string(prefix) + suffix), std::ios::binary);
            f.write(reinterpret_cast<const char *>(h.data()), h.size() * 4);
        }
    };
    dump(x, ".conv.f32");
    auto norm = b.view(x.rows, cfg.hidden, Type::F32),
         update = c.view(x.rows, cfg.hidden, Type::F32);
    for (const auto &l : layers_) {
        g.norm(x, l.pre, norm, cfg.eps, false, l.pre_bias);
        g.linear(norm, l.qkv, qkv.view(x.rows, (cfg.heads + 2 * cfg.kvheads) * cfg.dim, Type::F32),
                 true);
        g.qkv(qkv.view(x.rows, (cfg.heads + 2 * cfg.kvheads) * cfg.dim, Type::F32),
              q.view(x.rows, cfg.heads * cfg.dim, Type::F32),
              k.view(x.rows, cfg.kvheads * cfg.dim, Type::F32),
              v.view(x.rows, cfg.kvheads * cfg.dim, Type::F32), nullptr, 1, x.rows, 0, x.rows,
              cfg.heads, cfg.kvheads, cfg.dim, {}, {}, cfg.eps, false, rope_, 0);
        // HF Mimi eager/SDPA uses create_causal_mask (full causal), not the FA-only window.
        g.attention(q.view(x.rows, cfg.heads * cfg.dim, Type::F32),
                    k.view(x.rows, cfg.kvheads * cfg.dim, Type::F32),
                    v.view(x.rows, cfg.kvheads * cfg.dim, Type::F32),
                    att.view(x.rows, cfg.heads * cfg.dim, Type::F32), nullptr, 1, x.rows, 0, x.rows,
                    cfg.heads, cfg.kvheads, cfg.dim, true, 0, 0);
        g.linear(att.view(x.rows, cfg.heads * cfg.dim, Type::F32), l.out, update, true);
        g.add(x, update, x, l.attn_scale);
        g.norm(x, l.post, norm, cfg.eps, false, l.post_bias);
        auto hidden = ff.view(x.rows, cfg.intermediate, Type::F32);
        g.linear(norm, l.fc1, hidden, true);
        g.activation(hidden, hidden, 2);
        g.linear(hidden, l.fc2, update, true);
        g.add(x, update, x, l.mlp_scale);
    }
    dump(x, ".transformer.f32");
    auto down = x;
    if (downsample_) {
        down = b.view(downsample_->rows(x.rows), downsample_->out, Type::F32);
        downsample_->run(g, x, workspace(*downsample_, x.rows), down);
    }
    dump(down, ".latent.f32");
    auto r = residual.view(down.rows, config_.codebook_dim, Type::F32),
         sc = scores.view(down.rows, config_.codebook_size, Type::F32);
    for (int cb = 0; cb < config_.codebooks; cb++) {
        if (cb == 0 || cb == config_.semantic_codebooks)
            g.linear(down, cb ? acoustic_proj_ : semantic_proj_, r, true);
        g.linear(r, tables_[cb], sc, true);
        encoder_quantize(g.stream, r, sc, tables_[cb], norms_[cb], codes.ptr<int>(), cb,
                         config_.codebooks);
    }
    std::vector<int> result(size_t(down.rows) * config_.codebooks);
    g.download(result.data(), codes.data(), codes.size());
    return result;
}
} // namespace tts
