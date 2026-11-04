#include "support/profiler.h"
#include "ops/basic.h"
#include "models/mimi/encoder.h"
#include "layers/convolution.h"
#include "ops/vector_quantization.h"
#include <cmath>
#include <algorithm>
namespace tts {
AudioEncoder::Residual::Residual(Weights &w, const std::string &prefix, int i, int dilation)
    : first(w, prefix + ".encoder.layers." + std::to_string(i) + ".block.1.conv", CausalConvolution::Config{1, dilation}),
      second(w, prefix + ".encoder.layers." + std::to_string(i) + ".block.3.conv") {}
AudioEncoder::Stage::Stage(Weights &w, const std::string &prefix, int i, int ratio, const EncoderConfig &c)
    : down(w, prefix + ".encoder.layers." + std::to_string(i + c.residual_layers + 1) + ".conv", CausalConvolution::Config{ratio}) {
    int dilation = 1;
    for (int j = 0; j < c.residual_layers; ++j) {
        blocks.emplace_back(w, prefix, i + j, dilation);
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
AudioEncoder::AudioEncoder(Weights &w, const std::string &prefix, EncoderConfig config)
    : config_(std::move(config)), first_(w, prefix + ".encoder.layers.0.conv"),
      // Mimi's ModuleList: first conv, (residual blocks, ELU, downconv)*stages, ELU, last conv.
      last_(w, prefix + ".encoder.layers." +
                   std::to_string(2 + config_.ratios.size() * (config_.residual_layers + 2)) + ".conv"),
      rope_storage_({config_.transformer.max_positions, config_.transformer.dim}, Type::F32,
                    w.device()) {
    const auto &c = config_.transformer;
    if (first_.in != 1 || first_.out != config_.filters || first_.kernel != config_.first_kernel ||
        last_.out != c.hidden || last_.kernel != config_.last_kernel)
        throw std::invalid_argument("encoder boundary convolution weight/configuration mismatch");
    int index = 1, channels = first_.out;
    for (int ratio : config_.ratios) {
        stages_.emplace_back(w, prefix, index, ratio, config_);
        if (stages_.back().down.in != channels || stages_.back().down.out != 2 * channels)
            throw std::invalid_argument("encoder stage channel mismatch");
        channels = stages_.back().down.out;
        index += config_.residual_layers + 2;
    }
    if (last_.in != channels)
        throw std::invalid_argument("encoder last convolution channel mismatch");
    if (config_.downsample_stride != 1) {
        downsample_ = std::make_unique<CausalConvolution>(w, prefix + ".downsample.conv",
            CausalConvolution::Config{config_.downsample_stride, 1, false, false, true}, TensorView{});
        if (downsample_->in != c.hidden || downsample_->out != c.hidden ||
            downsample_->kernel != 2 * config_.downsample_stride)
            throw std::invalid_argument("encoder resampling weight/configuration mismatch");
    }
    auto matrix = [&](const std::string &name, int rows, int cols, bool upload = true) {
        w.validate(name, {rows, cols}, Type::F32);
        return upload ? w.get(name) : TensorView{};
    };
    auto vector = [&](const std::string &name) {
        w.validate(name, {c.hidden}, Type::F32);
        return w.get(name);
    };
    for (int i = 0; i < c.layers; i++) {
        std::string p = prefix + ".encoder_transformer.layers." + std::to_string(i);
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
        w.get(prefix + ".quantizer.semantic_residual_vector_quantizer.input_proj.weight");
    acoustic_proj_ =
        w.get(prefix + ".quantizer.acoustic_residual_vector_quantizer.input_proj.weight");
    for (auto projection : {semantic_proj_, acoustic_proj_})
        if (projection.type != Type::F32 || projection.rows != config_.codebook_dim ||
            projection.cols != c.hidden)
            throw std::invalid_argument("encoder quantizer projection shape");
    int entries = config_.codebook_size, dimension = config_.codebook_dim;
    for (int i = 0; i < config_.codebooks; i++) {
        std::string p =
            prefix + ".quantizer." +
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
    ops::rotary_table(rope_, c.dim, c.theta, 1, 0);
}
// Planning and allocation use the same shapes and alignment. No model-sized upper bounds.
struct AudioEncoder::Storage {
    Buffer a, b, c, projected, qkv, q, k, v, att, ff, residual, scores, codes;
    Storage(const AudioEncoder &model, int samples) {
        const auto &config = model.config_;
        const auto &tr = config.transformer;
        if (samples <= 0 ||
            samples / config.stride + (samples % config.stride != 0) > tr.max_positions)
            throw std::invalid_argument("reference audio is empty or exceeds encoder positions");
        size_t elements = size_t(samples) * model.first_.out, scratch = 0;
        auto conv = [&](const CausalConvolution &op, int rows) {
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
        a = Buffer(elements * sizeof(float));
        b = Buffer(a.size());
        c = Buffer(a.size());
        projected = Buffer(scratch * sizeof(float));
        qkv = Buffer(size_t(rows) * (tr.heads + 2 * tr.kvheads) * tr.dim * sizeof(float));
        q = Buffer(size_t(rows) * tr.heads * tr.dim * sizeof(float));
        k = Buffer(size_t(rows) * tr.kvheads * tr.dim * sizeof(float));
        v = Buffer(k.size());
        att = Buffer(q.size());
        ff = Buffer(size_t(rows) * tr.intermediate * sizeof(float));
        residual = Buffer(size_t(frames) * config.codebook_dim * sizeof(float));
        scores = Buffer(size_t(frames) * config.codebook_size * sizeof(float));
        codes = Buffer(size_t(frames) * config.codebooks * sizeof(int));
    }
};
size_t AudioEncoder::workspace_bytes(int samples) const {
    return Tensor::plan([&] { Storage storage(*this, samples); });
}
void AudioEncoder::prepare(ForwardContext &context, int max_samples) const {
    if (max_samples < 1) throw std::invalid_argument("invalid encoder planning capacity");
    size_t input_elements = 0, output_elements = 0;
    auto visit = [&](auto conv, auto matrix) {
        int rows = max_samples;
        for (const auto &stage : stages_) {
            for (const auto &block : stage.blocks) {
                conv(block.first, rows); conv(block.second, rows);
            }
            conv(stage.down, rows); rows = stage.down.rows(rows);
        }
        conv(last_, rows);
        for (const auto &layer : layers_)
            for (auto weight : {layer.qkv, layer.out, layer.fc1, layer.fc2}) matrix(weight, rows);
        if (downsample_) { conv(*downsample_, rows); rows = downsample_->rows(rows); }
        matrix(semantic_proj_, rows); matrix(acoustic_proj_, rows);
        for (auto table : tables_) matrix(table, rows);
    };
    visit([&](const CausalConvolution &conv, int) {
        input_elements = std::max(input_elements, size_t(conv.stride) * conv.in);
        output_elements = std::max(output_elements, size_t(conv.kernel) * conv.out);
    }, [&](TensorView weight, int) {
        input_elements = std::max(input_elements, size_t(weight.cols));
        output_elements = std::max(output_elements, size_t(weight.rows));
    });
    const int samples = std::min(max_samples, config_.stride);
    Buffer input(input_elements * sizeof(float));
    Buffer output(output_elements * sizeof(float));
    auto waveform = context.tensor({samples, 1}, Type::F32);
    if (context.mode == ForwardContext::Plan || !input.data()) {
        Tensor::reserve(workspace_bytes(samples));
        return;
    }
    visit([&](const CausalConvolution &conv, int rows) {
        conv.prepare(input.view(conv.stride, conv.in, Type::F32),
                        output.view(1, conv.kernel * conv.out, Type::F32), rows);
    }, [&](TensorView weight, int rows) {
        ops::prepare_matmul(input.view(1, weight.cols, Type::F32), weight,
                           output.view(1, weight.rows, Type::F32), true, 0, 1, 1, rows);
    });
    // A minimal legal path materializes reference-only kernel functions. All
    // large-M library plans were declared above; no full reference activation
    // buffers or real audio are needed during resource initialization.
    ops::zero(waveform);
    auto execution = context;
    execution.mode = ForwardContext::Execute;
    forward(waveform, execution);
}
TensorView AudioEncoder::forward(TensorView waveform, ForwardContext &context) const {
    Range range("tts.reference_encoder");
    if (waveform.type != Type::F32 || waveform.cols != 1)
        throw std::invalid_argument("reference waveform must contain mono F32 samples");
    const int samples = waveform.rows;
    if (context.mode == ForwardContext::Prepare) {
        prepare(context, samples);
        return TensorView::metadata(1, config_.codebooks, Type::I32);
    }
    const auto &cfg = config_.transformer;
    auto [a, b, c, projected, qkv, q, k, v, att, ff, residual, scores, codes] =
        Storage(*this, samples);
    auto output_codes = codes.view(int(codes.size() / sizeof(int) / config_.codebooks),
                                   config_.codebooks, Type::I32);
    if (context.mode != ForwardContext::Execute) return output_codes;
    const auto *projection_storage = &projected;
    auto workspace = [&](const CausalConvolution &cv, int n) {
        return projection_storage->view(cv.rows(n), cv.kernel * cv.out, Type::F32);
    };
    auto x = a.view(samples, first_.out, Type::F32);
    first_.forward(waveform, x, {});
    for (const auto &stage : stages_) {
        auto activated = b.view(x.rows, x.cols, Type::F32);
        ops::activation(x, activated, 3);
        for (size_t i = 0; i < stage.blocks.size(); ++i) {
            const auto &block = stage.blocks[i];
            auto mid = c.view(x.rows, block.first.out, Type::F32);
            block.first.forward(activated, mid, workspace(block.first, x.rows), true);
            // Keep the residual unactivated until all residual blocks have consumed it.
            bool last = i + 1 == stage.blocks.size();
            block.second.forward(mid, last ? activated : x, workspace(block.second, mid.rows), last, x);
            if (!last)
                ops::activation(x, activated, 3);
        }
        auto next = a.view(stage.down.rows(x.rows), stage.down.out, Type::F32);
        stage.down.forward(activated, next, workspace(stage.down, x.rows));
        x = next;
    }
    auto act = b.view(x.rows, x.cols, Type::F32);
    ops::activation(x, act, 3);
    x = a.view(act.rows, last_.out, Type::F32);
    last_.forward(act, x, workspace(last_, act.rows));
    auto norm = b.view(x.rows, cfg.hidden, Type::F32),
         update = c.view(x.rows, cfg.hidden, Type::F32);
    for (const auto &l : layers_) {
        ops::norm(x, l.pre, norm, cfg.eps, false, l.pre_bias);
        ops::matmul(norm, l.qkv, qkv.view(x.rows, (cfg.heads + 2 * cfg.kvheads) * cfg.dim, Type::F32),
                 true);
        ops::write_qkv(qkv.view(x.rows, (cfg.heads + 2 * cfg.kvheads) * cfg.dim, Type::F32),
              q.view(x.rows, cfg.heads * cfg.dim, Type::F32),
              k.view(x.rows, cfg.kvheads * cfg.dim, Type::F32),
              v.view(x.rows, cfg.kvheads * cfg.dim, Type::F32), nullptr, 1, x.rows, 0, x.rows,
              cfg.heads, cfg.kvheads, cfg.dim, {}, {}, cfg.eps, false, rope_, 0);
        // HF Mimi eager/SDPA uses create_causal_mask (full causal), not the FA-only window.
        ops::attention(q.view(x.rows, cfg.heads * cfg.dim, Type::F32),
                    k.view(x.rows, cfg.kvheads * cfg.dim, Type::F32),
                    v.view(x.rows, cfg.kvheads * cfg.dim, Type::F32),
                    att.view(x.rows, cfg.heads * cfg.dim, Type::F32), nullptr, 1, x.rows, 0, x.rows,
                    cfg.heads, cfg.kvheads, cfg.dim, true, 0, 0);
        ops::matmul(att.view(x.rows, cfg.heads * cfg.dim, Type::F32), l.out, update, true);
        ops::add(x, update, x, l.attn_scale);
        ops::norm(x, l.post, norm, cfg.eps, false, l.post_bias);
        auto hidden = ff.view(x.rows, cfg.intermediate, Type::F32);
        ops::matmul(norm, l.fc1, hidden, true);
        ops::activation(hidden, hidden, 2);
        ops::matmul(hidden, l.fc2, update, true);
        ops::add(x, update, x, l.mlp_scale);
    }
    auto down = x;
    if (downsample_) {
        down = b.view(downsample_->rows(x.rows), downsample_->out, Type::F32);
        downsample_->forward(x, down, workspace(*downsample_, x.rows));
    }
    auto r = residual.view(down.rows, config_.codebook_dim, Type::F32),
         sc = scores.view(down.rows, config_.codebook_size, Type::F32);
    for (int cb = 0; cb < config_.codebooks; cb++) {
        if (cb == 0 || cb == config_.semantic_codebooks)
            ops::matmul(down, cb ? acoustic_proj_ : semantic_proj_, r, true);
        ops::matmul(r, tables_[cb], sc, true);
        vector_quantize( r, sc, tables_[cb], norms_[cb], codes.ptr<int>(), cb,
                         config_.codebooks);
    }
    return output_codes;
}
} // namespace tts
