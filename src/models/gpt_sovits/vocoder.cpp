#include "ops/basic.h"
#include "models/gpt_sovits/vocoder.h"
#include <numeric>
#include <tuple>

namespace tts {
TensorView Vocoder::forward(const TensorBatch &input, TensorView layout, ForwardContext &context) const {
    const auto type = pre_.type();
    const int stride = samples(context.max_time);
    auto output = context.tensor({context.max_batch * stride, 1}, type);
    if (context.mode == ForwardContext::Execute &&
        (layout.device != -1 || layout.type != Type::I32 || layout.cols != 2 || layout.rows < input.rows))
        throw std::invalid_argument("audio output layout must contain host offset/count pairs");
    if (context.mode == ForwardContext::Capture) {
        const int core = int(context.parameters["core_frames"]);
        const int future = int(context.parameters["future_frames"]);
        const int left = int(context.parameters["left_frames"]);
        std::vector<int> times{core + future, left + core + future, left + core};
        std::sort(times.begin(), times.end(), std::greater<int>());
        times.erase(std::unique(times.begin(), times.end()), times.end());
        for (size_t index = context.batches.size; index; --index)
            for (int time : times) {
                if (time > context.max_time) continue;
                Tensor::Scope scratch;
                const int batch = context.batches[index - 1];
                auto invocation = context;
                invocation.max_batch = batch;
                invocation.max_time = time;
                invocation.max_rows = batch * time;
                invocation.batch = ForwardBatch(batch, time);
                forward(input.chunks.front().values.slice(0, batch * time),
                    output.slice(0, batch * samples(time)), invocation);
            }
        return output;
    }
    int written = 0;
    for (const auto &chunk : input.chunks) {
        Tensor::Scope scratch;
        int first = 0, end = chunk.time, offset = 0, count = chunk.time;
        if (chunk.frame_count) {
            count = std::min(chunk.frame_count,
                std::max(1, context.max_time - (tile_capacity(1) - 1)));
            std::tie(first, end) = input_frames(chunk.first_frame, count, chunk.time);
            offset = chunk.first_frame - first;
        }
        auto chunk_context = context;
        chunk_context.max_batch = int(chunk.rows.size());
        chunk_context.max_time = end - first;
        chunk_context.max_rows = chunk_context.max_batch * chunk_context.max_time;
        chunk_context.batch = ForwardBatch(chunk_context.max_batch, chunk_context.max_time);
        const int stride = samples(chunk_context.max_time), total = stride * chunk_context.max_batch;
        if (total > output.rows - written) throw std::invalid_argument("audio chunks exceed their declared output capacity");
        forward(chunk.values.slice(first, chunk_context.max_rows), output.slice(written, total), chunk_context);
        if (context.mode == ForwardContext::Execute)
            for (size_t row = 0; row < chunk.rows.size(); ++row) {
                const int target = chunk.rows[row];
                if (target < 0 || target >= layout.rows) throw std::invalid_argument("invalid audio chunk membership");
                layout.ptr<int>()[size_t(target) * 2] = written + int(row) * stride + samples(offset);
                layout.ptr<int>()[size_t(target) * 2 + 1] = samples(count);
            }
        written += total;
    }
    return output.slice(0, written);
}
TensorView Vocoder::forward(TensorView mel, TensorView destination, ForwardContext &context) const {
    if (context.mode == ForwardContext::Prepare) {
        Workspace workspace(*this, 1, 1);
        if (workspace.activations[0].data()) prepare(workspace, context.max_batch, context.max_time);
    } else {
        Workspace workspace(*this, context.max_batch, context.max_time);
        if (context.mode == ForwardContext::Execute || context.mode == ForwardContext::Capture)
            context.run(this, "vocoder", {context.max_batch, context.max_time},
                {mel.data, destination.data}, [&] { forward(workspace, mel, destination, context.batch); });
    }
    return destination;
}
static ops::ConvGeometry conv_geometry(Weights &w, const std::string &name, int stride,
                                       int dilation = 1, bool transpose = false) {
    const auto &s = w.entry(name + ".weight").shape;
    if (s.size() != 3)
        throw std::invalid_argument("vocoder convolution must have rank 3");
    return {transpose ? s[0] : s[1],
            transpose ? s[1] : s[0],
            s[2],
            stride,
            transpose ? (s[2] - stride) / 2 : dilation * (s[2] - 1) / 2,
            dilation,
            1,
            transpose};
}
Vocoder::Vocoder(Weights &w, const std::string &name, const Json &config)
    : pre_(w, name + ".conv_pre", conv_geometry(w, name + ".conv_pre", 1)),
      post_(w, name + ".conv_post", conv_geometry(w, name + ".conv_post", 1)) {
    const auto rates = config.at("upsample_rates").get<std::vector<int>>();
    const auto dilations = config.at("resblock_dilations").get<std::vector<std::vector<int>>>();
    if (rates.empty() || dilations.empty())
        throw std::invalid_argument("empty vocoder structure");
    int channels = pre_.geometry().output_channels;
    for (size_t i = 0; i < rates.size(); ++i) {
        auto prefix = name + ".ups." + std::to_string(i);
        Stage stage{Convolution(w, prefix, conv_geometry(w, prefix, rates[i], 1, true)), {}};
        if (stage.upsample.geometry().input_channels != channels)
            throw std::invalid_argument("disconnected vocoder stage");
        channels = stage.upsample.geometry().output_channels;
        for (size_t j = 0; j < dilations.size(); ++j) {
            ResidualBlock block;
            if (dilations[j].empty())
                throw std::invalid_argument("empty vocoder residual block");
            auto base = name + ".resblocks." + std::to_string(i * dilations.size() + j);
            for (size_t k = 0; k < dilations[j].size(); ++k) {
                auto a = base + ".convs1." + std::to_string(k);
                auto b = base + ".convs2." + std::to_string(k);
                block.first.emplace_back(w, a, conv_geometry(w, a, 1, dilations[j][k]));
                block.second.emplace_back(w, b, conv_geometry(w, b, 1));
                for (auto *layer : {&block.first.back(), &block.second.back()})
                    if (layer->geometry().input_channels != channels ||
                        layer->geometry().output_channels != channels ||
                        layer->type() != pre_.type())
                        throw std::invalid_argument("vocoder residual shape/type mismatch");
            }
            stage.residuals.push_back(std::move(block));
        }
        stages_.push_back(std::move(stage));
    }
    if (post_.geometry().input_channels != channels || post_.geometry().output_channels != 1 ||
        samples(2) != 2 * samples(1))
        throw std::invalid_argument("vocoder output shape mismatch");
}
int Vocoder::samples(int frames) const {
    if (!frames)
        return 0;
    for (const auto &stage : stages_)
        frames = stage.upsample.geometry().output_length(frames);
    return frames;
}
std::pair<int64_t, int64_t> Vocoder::input_interval(int64_t begin, int64_t end) const {
    using Span = std::pair<int64_t, int64_t>;
    auto inverse = [](const Convolution &c, Span span) {
        return ops::convolution_input_span(c.geometry(), span);
    };
    auto unite = [](Span a, Span b) {
        return Span{std::min(a.first, b.first), std::max(a.second, b.second)};
    };
    Span span = inverse(post_, {begin, end});
    for (auto stage = stages_.rbegin(); stage != stages_.rend(); ++stage) {
        Span required = span;
        for (const auto &block : stage->residuals) {
            Span branch = span;
            for (size_t i = block.first.size(); i-- > 0;)
                branch = unite(branch, inverse(block.first[i], inverse(block.second[i], branch)));
            required = unite(required, branch);
        }
        span = inverse(stage->upsample, required);
    }
    return inverse(pre_, span);
}
std::pair<int, int> Vocoder::input_frames(int first, int count, int total) const {
    if (first < 0 || count < 1 || total < count || first > total - count)
        throw std::invalid_argument("invalid vocoder output tile");
    auto span = input_interval(samples(first), samples(first + count));
    return {int(std::max<int64_t>(0, span.first)), int(std::min<int64_t>(total, span.second))};
}
int Vocoder::tile_capacity(int core_frames) const {
    auto span = input_interval(0, samples(core_frames));
    int64_t size = span.second - span.first;
    if (core_frames < 1 || size < 1 || size > INT32_MAX)
        throw std::invalid_argument("invalid vocoder tile capacity");
    return int(size);
}
Vocoder::Workspace::Workspace(const Vocoder &v, int batch, int frames)
    : max_batch(batch), max_frames(frames) {
    if (batch < 1 || frames < 1)
        throw std::invalid_argument("invalid vocoder capacity");
    size_t activation_bytes = 0, projection_bytes = 0;
    auto visit = [&](const Convolution &c, int input) {
        int output = c.geometry().output_length(input);
        activation_bytes =
            std::max(activation_bytes,
                     size_t(batch) * output * c.geometry().output_channels * bytes(c.type()));
        projection_bytes = std::max(projection_bytes, c.workspace_bytes(batch, input));
        return output;
    };
    int time = visit(v.pre_, frames);
    for (const auto &s : v.stages_) {
        time = visit(s.upsample, time);
        for (const auto &b : s.residuals)
            for (size_t i = 0; i < b.first.size(); ++i) {
                visit(b.first[i], time);
                visit(b.second[i], time);
            }
    }
    visit(v.post_, time);
    for (auto &a : activations)
        a = Buffer(activation_bytes);
    projection = Buffer(projection_bytes);
}
void Vocoder::prepare(Workspace &w, int batch, int frames) const {
    if (batch < 1 || frames < 1)
        throw std::invalid_argument("invalid vocoder planning capacity");
    auto visit = [&](const Convolution &c, int input) {
        if (int64_t(batch) * input > INT32_MAX)
            throw std::overflow_error("vocoder planning row count");
        if (c.workspace_bytes(1, 1))
            c.prepare(w.activations[0].view(1, c.geometry().input_channels, c.type()),
                      w.projection.view(1, c.geometry().kernel * c.geometry().output_channels,
                                        Type::F32), batch * input);
        return c.geometry().output_length(input);
    };
    int time = visit(pre_, frames);
    for (const auto &stage : stages_) {
        time = visit(stage.upsample, time);
        for (const auto &block : stage.residuals)
            for (size_t i = 0; i < block.first.size(); ++i) {
                visit(block.first[i], time);
                visit(block.second[i], time);
            }
    }
    visit(post_, time);
}
void Vocoder::forward(Workspace &w, TensorView mel, TensorView audio, const ForwardBatch &input) const {

    int batch = input.batch(), frames = input.time();
    if (batch < 1 || batch > w.max_batch || frames < 1 || frames > w.max_frames ||
        mel.rows != batch * frames || mel.cols != pre_.geometry().input_channels ||
        audio.rows != batch * samples(frames) || audio.cols != 1)
        throw std::invalid_argument("vocoder request exceeds initialized workspace");
    Type type = pre_.type();
    auto view = [&](int slot, int time, int channels) {
        return w.activations[slot].view(batch * time, channels, type);
    };
    auto run = [&](const Convolution &c, TensorView x, TensorView y, int time, ops::Epilogue e) {
        c.forward(x, y, ForwardBatch{batch, time},
                  w.projection.view(batch * time,
                                    c.geometry().kernel * c.geometry().output_channels, Type::F32),
                  e);
    };
    int time = frames, channels = pre_.geometry().output_channels;
    ops::Epilogue e;
    e.activation = ops::Activation::LeakyRelu;
    e.raw = view(0, time, channels);
    run(pre_, mel, view(1, time, channels), time, e);
    int input_slot = 1, sum_slot = 5;
    for (size_t index = 0; index < stages_.size(); ++index) {
        const auto &s = stages_[index];
        int next = s.upsample.geometry().output_length(time);
        int output = s.upsample.geometry().output_channels;
        e.raw = view(0, next, output);
        run(s.upsample, view(input_slot, time, channels), view(input_slot, next, output), time, e);
        time = next;
        channels = output;
        auto initial = view(0, time, channels), initial_act = view(input_slot, time, channels);
        auto temp = view(2, time, channels), residual = view(3, time, channels);
        auto activated = view(4, time, channels), accumulated = view(sum_slot, time, channels);
        for (size_t branch = 0; branch < s.residuals.size(); ++branch) {
            const auto &b = s.residuals[branch];
            for (size_t layer = 0; layer < b.first.size(); ++layer) {
                e = {};
                e.activation = ops::Activation::LeakyRelu;
                run(b.first[layer], layer ? activated : initial_act, temp, time, e);
                e.residual = layer ? residual : initial;
                e.raw = residual;
                if (layer + 1 == b.first.size()) {
                    e.activation = ops::Activation::Identity;
                    e.raw = {};
                    run(b.second[layer], temp, branch ? residual : accumulated, time, e);
                } else
                    run(b.second[layer], temp, activated, time, e);
            }
            if (branch) {
                e = {};
                e.residual = accumulated;
                if (branch + 1 == s.residuals.size()) {
                    e.output_scale = 1.f / s.residuals.size();
                    e.activation = ops::Activation::LeakyRelu;
                    e.alpha = index + 1 == stages_.size() ? .01f : .1f;
                }
                ops::pointwise( residual, accumulated, e);
            }
        }
        // A single-branch configuration still needs its activation epilogue.
        if (s.residuals.size() == 1) {
            e = {};
            e.activation = ops::Activation::LeakyRelu;
            e.alpha = index + 1 == stages_.size() ? .01f : .1f;
            ops::pointwise( accumulated, accumulated, e);
        }
        // The next layer reads this view directly; no activation copy is needed.
        if (index + 1 == stages_.size()) {
            e = {};
            e.activation = ops::Activation::Tanh;
            run(post_, accumulated, audio, time, e);
        } else {
            // Ping-pong ownership of the already activated backing storage.
            std::swap(input_slot, sum_slot);
            e = {};
            e.activation = ops::Activation::LeakyRelu;
        }
    }
}
} // namespace tts
