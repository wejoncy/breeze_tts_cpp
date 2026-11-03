#include "models/gpt_sovits/wavenet.h"
namespace tts::gpt_sovits {
WaveNet::WaveNet(Weights &w, const std::string &prefix, int layers, int dilation_rate)
    : pre_(w, prefix + ".pre"), conditioning_(w, prefix + ".enc.cond_layer"),
      post_(w, prefix + ".proj") {
    int hidden = pre_.output_channels();
    if (layers < 1 || dilation_rate < 1 || conditioning_.output_channels() != 2 * hidden * layers)
        throw std::invalid_argument("WaveNet conditioning structure mismatch");
    int dilation = 1;
    for (int i = 0; i < layers; ++i) {
        auto p = prefix + ".enc.in_layers." + std::to_string(i);
        const auto &s = w.entry(p + ".weight").shape;
        if (s.size() != 3 || s[0] != 2 * hidden || s[1] != hidden || s[2] % 2 != 1)
            throw std::invalid_argument("WaveNet checkpoint geometry");
        layers_.emplace_back(w, p,
                             ops::ConvGeometry{hidden, 2 * hidden, s[2], 1,
                                               dilation * (s[2] - 1) / 2, dilation, 1, false});
        residuals_.emplace_back(w, prefix + ".enc.res_skip_layers." + std::to_string(i));
        if (residuals_.back().output_channels() != hidden * (i + 1 == layers ? 1 : 2))
            throw std::invalid_argument("WaveNet residual checkpoint geometry");
        if (dilation > INT32_MAX / dilation_rate)
            throw std::overflow_error("WaveNet dilation");
        dilation *= dilation_rate;
    }
}
WaveNet::Workspace::Workspace(const WaveNet &n, int batch, int time, BufferAllocator *allocator, int sequences)
    : max_batch(sequences ? sequences : batch), max_time(time), max_rows(int64_t(batch) * time) {
    if (batch < 1 || time < 1 || int64_t(batch) * time > INT32_MAX)
        throw std::invalid_argument("invalid WaveNet workspace");
    int width = n.pre_.output_channels();
    Type type = n.pre_.activation_type();
    size_t rows = size_t(batch) * time;
    hidden = Buffer(rows * width * bytes(type), allocator);
    skip = Buffer(hidden.size(), allocator);
    activated = Buffer(hidden.size(), allocator);
    gates = Buffer(hidden.size() * 2, allocator);
    projected = Buffer(gates.size(), allocator);
    condition = Buffer(size_t(max_batch) * n.conditioning_.output_channels() * bytes(type), allocator);
    size_t scratch = 0;
    for (const auto &c : n.layers_)
        scratch = std::max(scratch, c.workspace_bytes(batch, time));
    convolution = Buffer(scratch, allocator);
}
void WaveNet::prepare(Workspace &w, int rows) const {
    int hidden = pre_.output_channels();
    Type type = pre_.activation_type();
    pre_.prepare_range(w.gates.view(1, pre_.input_channels(), type), w.hidden.view(1, hidden, type), rows);
    conditioning_.prepare_range(w.gates.view(1, conditioning_.input_channels(), type),
                                 w.condition.view(1, conditioning_.output_channels(), type), w.max_batch);
    for (size_t i = 0; i < layers_.size(); ++i) {
        const auto &c = layers_[i];
        c.prepare(w.hidden.view(1, hidden, type),
                  w.convolution.view(1, c.geometry().kernel * 2 * hidden, Type::F32), rows);
        residuals_[i].prepare_range(w.activated.view(1, hidden, type),
                                    w.projected.view(1, residuals_[i].output_channels(), type), rows);
    }
    post_.prepare_range(w.skip.view(1, hidden, type),
                         w.gates.view(1, post_.output_channels(), type), rows);
}
void WaveNet::forward(Workspace &w, TensorView input, TensorView style,
                      TensorView output, const ForwardBatch &batch_input, int valid_time,
                      const void *const *condition_sources,
                      void *const *output_rows) const {
    auto &gpu = backend_for(input);
    int batch = batch_input.batch(), time = batch_input.time();
    const auto *packed = batch_input.sequence.addressing();
    Type type = pre_.activation_type();
    int hidden = pre_.output_channels();
    if (batch < 1 || (!packed && batch > w.max_batch) || time < 1 || time > w.max_time ||
        input.rows != (packed ? packed->rows : int64_t(batch) * time) ||
        input.rows > w.max_rows ||
        (!condition_sources && style.rows != batch) || input.type != type ||
        output.rows != input.rows || output.cols != post_.output_channels() || output.type != type ||
        valid_time < 0 || valid_time > time)
        throw std::invalid_argument("WaveNet input exceeds initialized workspace");
    int rows = input.rows;
    auto x = w.hidden.view(rows, hidden, type), skip = w.skip.view(rows, hidden, type);
    auto gates = w.gates.view(rows, 2 * hidden, type),
         activated = w.activated.view(rows, hidden, type);
    auto condition = condition_sources ? TensorView{}
        : w.condition.view(batch, conditioning_.output_channels(), type);
    ops::Epilogue mask;
    mask.rows_per_sequence = packed ? 1 : time;
    mask.packed = packed;
    mask.valid_rows_per_sequence = valid_time < time ? valid_time : 0;
    pre_.forward(input, x, mask);
    if (!condition_sources) conditioning_.forward(style, condition);
    for (size_t i = 0; i < layers_.size(); ++i) {
        const auto &c = layers_[i];
        c.forward(x, gates, batch_input,
                  w.convolution.view(rows, c.geometry().kernel * 2 * hidden, Type::F32));
        ops::gated(gpu.stream, gates, activated, ops::Activation::Tanh, time,
                   condition_sources ? TensorView{} : condition.columns(int(i) * 2 * hidden, 2 * hidden),
                   {}, packed, condition_sources, int(i) * 2 * hidden);
        bool last = i + 1 == layers_.size();
        auto projected = w.projected.view(rows, residuals_[i].output_channels(), type);
        residuals_[i].forward(activated, projected);
        ops::residual_split(gpu.stream, projected, x, skip, i == 0, last, time,
                            mask.valid_rows_per_sequence, packed);
    }
    mask.output_rows = output_rows;
    post_.forward(skip, output, mask);
}
} // namespace tts::gpt_sovits
