#include "audio_encoder.h"
#include <filesystem>
#include "encoder_kernels.h"
#include <cmath>
#include <fstream>
namespace tts {
EncoderConv::EncoderConv(Weights &w, const std::string &p, int s, bool rep)
    : stride(s), replicate(rep) {
    const auto &e = w.entry(p + ".conv.weight");
    if (e.type != Type::F32 || e.shape.size() != 3 || e.shape[2] % stride)
        throw std::runtime_error("unsupported encoder convolution");
    out = e.shape[0];
    in = e.shape[1];
    kernel = e.shape[2];
    bias = w.optional(p + ".conv.bias");
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
    int n = (x.rows + stride - 1) / stride, taps = kernel / stride;
    if (x.cols != in || y.rows != n || y.cols != out)
        throw std::runtime_error("encoder convolution shape");
    auto p = scratch.reshape({int(scratch.numel() / (kernel * out)), kernel * out}).slice(0, n);
    for (int phase = 0; phase < stride && phase < x.rows; phase++) {
        int count = (x.rows - 1 - phase) / stride + 1;
        g.linear(x.every_rows(phase, stride, count), weight.slice(phase * taps * out, taps * out),
                 p.columns(phase * taps * out, taps * out).slice(0, count), true);
    }
    encoder_conv_finish(g.stream, x, p, weight, bias, y, kernel, stride, replicate, elu, residual);
}
AudioEncoder::Stage::Stage(Weights &w, int i, int ratio)
    : first(w, "codec.encoder.encoder.layers." + std::to_string(i) + ".block.1"),
      second(w, "codec.encoder.encoder.layers." + std::to_string(i) + ".block.3"),
      down(w, "codec.encoder.encoder.layers." + std::to_string(i + 2), ratio) {}
AudioEncoder::AudioEncoder(Weights &w, int max_positions)
    : first_(w, "codec.encoder.encoder.layers.0"), last_(w, "codec.encoder.encoder.layers.14"),
      downsample_(w, "codec.encoder.downsample", 2, true),
      rope_storage_({max_positions, 64}, Type::F32, &w.backend().memory) {
    int ratios[] = {4, 5, 6, 8};
    for (int i = 0; i < 4; i++)
        stages_.emplace_back(w, 1 + 3 * i, ratios[i]);
    for (int i = 0; i < 8; i++) {
        std::string p = "codec.encoder.encoder_transformer.layers." + std::to_string(i);
        Layer l;
        l.pre = w.get(p + ".input_layernorm.weight");
        l.pre_bias = w.get(p + ".input_layernorm.bias");
        l.post = w.get(p + ".post_attention_layernorm.weight");
        l.post_bias = w.get(p + ".post_attention_layernorm.bias");
        l.qkv = w.fused(p + ".qkv", {p + ".self_attn.q_proj.weight", p + ".self_attn.k_proj.weight",
                                     p + ".self_attn.v_proj.weight"});
        l.out = w.get(p + ".self_attn.o_proj.weight");
        l.fc1 = w.get(p + ".mlp.fc1.weight");
        l.fc2 = w.get(p + ".mlp.fc2.weight");
        l.attn_scale = w.get(p + ".self_attn_layer_scale.scale");
        l.mlp_scale = w.get(p + ".mlp_layer_scale.scale");
        layers_.push_back(l);
    }
    semantic_proj_ =
        w.get("codec.encoder.quantizer.semantic_residual_vector_quantizer.input_proj.weight");
    acoustic_proj_ =
        w.get("codec.encoder.quantizer.acoustic_residual_vector_quantizer.input_proj.weight");
    for (int i = 0; i < 16; i++) {
        std::string p = "codec.encoder.quantizer." + std::string(i ? "acoustic" : "semantic") +
                        "_residual_vector_quantizer.layers." + std::to_string(i ? i - 1 : 0) +
                        ".codebook.";
        const auto &e = w.entry(p + "embed_sum");
        const auto &usage = w.entry(p + "cluster_usage");
        if (e.shape != std::vector<int>({2048, 256}) || e.type != Type::F32 ||
            usage.shape != std::vector<int>({2048}))
            throw std::runtime_error("encoder codebook shape");
        const auto *src = static_cast<const float *>(e.data);
        const auto *u = static_cast<const float *>(usage.data);
        std::vector<float> table(2048 * 256), norm(2048);
        for (int k = 0; k < 2048; k++)
            for (int d = 0; d < 256; d++) {
                float v = src[k * 256 + d] / std::max(u[k], 1e-5f);
                table[k * 256 + d] = v;
                norm[k] += v * v;
            }
        tables_.push_back(w.custom(p + "normalized", table.data(), 2048, 256, Type::F32));
        norms_.push_back(w.custom(p + "norm", norm.data(), 1, 2048, Type::F32));
    }
    rope_ = rope_storage_.view();
    w.backend().make_rope(rope_, 64, 10000, 1, 0);
}
namespace {
// Count and allocate the identical layout, including per-buffer alignment.
// Convolution temporaries retain validated model-specific channel bounds.
struct EncoderStorage {
    Buffer input, a, b, c, projected, qkv, q, k, v, att, ff, residual, scores, codes;
    EncoderStorage(int samples, BufferAllocator *arena) {
        if (samples <= 0)
            throw std::invalid_argument("reference audio is empty");
        size_t rounded_samples = (size_t(samples) + 1919) / 1920 * 1920;
        size_t rows = (size_t(samples) + 959) / 960, frames = (rows + 1) / 2;
        input = Buffer(size_t(samples) * sizeof(float), arena);
        a = Buffer(rounded_samples * 64 * sizeof(float), arena);
        b = Buffer(a.size(), arena);
        c = Buffer(a.size(), arena);
        projected = Buffer(rounded_samples * 256 * sizeof(float), arena);
        qkv = Buffer(rows * 1536 * sizeof(float), arena);
        q = Buffer(rows * 512 * sizeof(float), arena);
        k = Buffer(q.size(), arena);
        v = Buffer(q.size(), arena);
        att = Buffer(q.size(), arena);
        ff = Buffer(rows * 2048 * sizeof(float), arena);
        residual = Buffer(frames * 256 * sizeof(float), arena);
        scores = Buffer(frames * 2048 * sizeof(float), arena);
        codes = Buffer(frames * 16 * sizeof(int), arena);
    }
};
} // namespace
size_t AudioEncoder::workspace_bytes(int samples) {
    WorkspacePlan plan;
    EncoderStorage storage(samples, &plan);
    return plan.size;
}
std::vector<int> AudioEncoder::encode(Backend &g, BufferAllocator &arena,
                                      const std::vector<float> &mono) const {
    Range range("tts.reference_encoder");
    int samples = mono.size(), max_rows = (samples + 959) / 960;
    if (!samples || max_rows > rope_.rows)
        throw std::runtime_error("reference audio exceeds encoder position limit");
    auto [input, a, b, c, projected, qkv, q, k, v, att, ff, residual, scores, codes] =
        EncoderStorage(samples, &arena);
    g.upload(input.data(), mono.data(), input.size());
    auto x = a.view(samples, 64, Type::F32);
    encoder_first_conv(g.stream, input.view(samples, 1, Type::F32), first_.weight, first_.bias, x, 7);
    for (const auto &stage : stages_) {
        auto activated = b.view(x.rows, x.cols, Type::F32);
        g.activation(x, activated, 3);
        auto mid = c.view(x.rows, stage.first.out, Type::F32);
        auto workspace = [&](const EncoderConv &cv, int n) {
            return projected.view((n + cv.stride - 1) / cv.stride, cv.kernel * cv.out, Type::F32);
        };
        stage.first.run(g, activated, workspace(stage.first, x.rows), mid, true);
        stage.second.run(g, mid, workspace(stage.second, mid.rows), activated, true, x);
        auto next =
            a.view((x.rows + stage.down.stride - 1) / stage.down.stride, stage.down.out, Type::F32);
        stage.down.run(g, activated, workspace(stage.down, x.rows), next);
        x = next;
    }
    auto act = b.view(x.rows, x.cols, Type::F32);
    g.activation(x, act, 3);
    x = a.view(act.rows, 512, Type::F32);
    last_.run(g, act, projected.view(x.rows, last_.kernel * 512, Type::F32), x);
    auto dump = [&](TensorView value, const char *suffix) {
        if (const char *prefix = std::getenv("TTS_ENCODER_DUMP")) {
            std::vector<float> h(value.numel());
            g.download(h.data(), value.data, value.size());
            std::ofstream f(std::filesystem::u8path(std::string(prefix) + suffix), std::ios::binary);
            f.write(reinterpret_cast<const char *>(h.data()), h.size() * 4);
        }
    };
    dump(x, ".conv.f32");
    auto norm = b.view(x.rows, 512, Type::F32), update = c.view(x.rows, 512, Type::F32);
    for (const auto &l : layers_) {
        g.norm(x, l.pre, norm, 1e-5f, false, l.pre_bias);
        g.linear(norm, l.qkv, qkv.view(x.rows, 1536, Type::F32), true);
        g.qkv(qkv.view(x.rows, 1536, Type::F32), q.view(x.rows, 512, Type::F32),
              k.view(x.rows, 512, Type::F32), v.view(x.rows, 512, Type::F32), nullptr, 1, x.rows, 0,
              x.rows, 8, 8, 64, {}, {}, 1e-5f, false, rope_, 0);
        // HF Mimi eager/SDPA uses create_causal_mask (full causal), not the FA-only window.
        g.attention(q.view(x.rows, 512, Type::F32), k.view(x.rows, 512, Type::F32),
                    v.view(x.rows, 512, Type::F32), att.view(x.rows, 512, Type::F32), nullptr, 1,
                    x.rows, 0, x.rows, 8, 8, 64, true, 0, 0);
        g.linear(att.view(x.rows, 512, Type::F32), l.out, update, true);
        g.add(x, update, x, l.attn_scale);
        g.norm(x, l.post, norm, 1e-5f, false, l.post_bias);
        auto hidden = ff.view(x.rows, 2048, Type::F32);
        g.linear(norm, l.fc1, hidden, true);
        g.activation(hidden, hidden, 2);
        g.linear(hidden, l.fc2, update, true);
        g.add(x, update, x, l.mlp_scale);
    }
    dump(x, ".transformer.f32");
    auto down = b.view((x.rows + 1) / 2, 512, Type::F32);
    downsample_.run(g, x, projected.view(down.rows, 2048, Type::F32), down);
    dump(down, ".latent.f32");
    auto r = residual.view(down.rows, 256, Type::F32), sc = scores.view(down.rows, 2048, Type::F32);
    for (int cb = 0; cb < 16; cb++) {
        if (cb <= 1)
            g.linear(down, cb ? acoustic_proj_ : semantic_proj_, r, true);
        g.linear(r, tables_[cb], sc, true);
        encoder_quantize(g.stream, r, sc, tables_[cb], norms_[cb], codes.ptr<int>(), cb);
    }
    std::vector<int> result(size_t(down.rows) * 16);
    g.download(result.data(), codes.data(), codes.size());
    return result;
}
} // namespace tts
