#include "ops/basic.h"
#include "layers/audio/wavenet.h"
#include "core/forward_context.h"
namespace tts {
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
WaveNet::Workspace::Workspace(const WaveNet &network, const ForwardContext &context)
    : max_batch(context.max_batch), max_time(context.max_time), max_rows(context.max_rows) {
    if (max_batch < 1 || max_time < 1 || max_rows < 1)
        throw std::invalid_argument("invalid WaveNet workspace");
    const int width = network.pre_.output_channels();
    const auto type = network.pre_.activation_type();
    hidden = Buffer(size_t(max_rows) * width * bytes(type));
    skip = Buffer(hidden.size());
    activated = Buffer(hidden.size());
    gates = Buffer(hidden.size() * 2);
    projected = Buffer(gates.size());
    size_t scratch_bytes = 0;
    for (const auto &convolution : network.layers_)
        scratch_bytes = std::max(scratch_bytes, convolution.workspace_bytes(1, max_rows));
    convolution = Buffer(scratch_bytes);
}
TensorView WaveNet::condition(TensorView style, ForwardContext &context) const {
    auto parameters = context;
    parameters.max_rows = context.max_batch;
    return conditioning_.forward(style, parameters);
}
void WaveNet::forward(TensorView input, TensorView condition, TensorView output,
                       ForwardContext &context, TensorView output_rows) const {
    if (output_rows.numel() && (output_rows.type != Type::I64 || output_rows.cols != 1 ||
                                output_rows.rows != input.rows || output_rows.device != input.device))
        throw std::invalid_argument("WaveNet output row map shape/type mismatch");
    Tensor::Scope scratch;
    Workspace workspace(*this, context);
    if (context.mode == ForwardContext::Prepare && input.data)
        prepare(workspace, context.max_rows);
    else if (context.mode == ForwardContext::Execute)
        forward(workspace, input, condition, output, context.batch, output_rows.ptr<void *>());
}
void WaveNet::prepare(Workspace &w, int rows) const {
    int hidden = pre_.output_channels();
    Type type = pre_.activation_type();
    pre_.prepare_range(w.gates.view(1, pre_.input_channels(), type), w.hidden.view(1, hidden, type), rows);
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
void WaveNet::forward(Workspace &w, TensorView input, TensorView condition,
                      TensorView output, const ForwardBatch &batch_input, void *const *output_rows) const {

    int batch = batch_input.batch(), time = batch_input.time();
    const auto *packed = batch_input.sequence.addressing();
    Type type = pre_.activation_type();
    int hidden = pre_.output_channels();
    const bool indirect = condition.type == Type::I64;
    const auto *condition_sources = indirect ? condition.ptr<const void *>() : nullptr;
    if (batch < 1 || (!packed && batch > w.max_batch) || time < 1 || time > w.max_time ||
        input.rows != (packed ? packed->rows : int64_t(batch) * time) ||
        input.rows > w.max_rows ||
        !condition.data || condition.device != input.device || condition.rows != batch || (indirect ? condition.cols != 1
            : condition.type != type || condition.cols != condition_channels()) || input.type != type ||
        output.rows != input.rows || output.cols != post_.output_channels() || output.type != type)
        throw std::invalid_argument("WaveNet input exceeds initialized workspace");
    int rows = input.rows;
    auto x = w.hidden.view(rows, hidden, type), skip = w.skip.view(rows, hidden, type);
    auto gates = w.gates.view(rows, 2 * hidden, type),
         activated = w.activated.view(rows, hidden, type);
    ops::Epilogue mask;
    mask.rows_per_sequence = packed ? 1 : time;
    mask.packed = packed;
    pre_.forward(input, x, mask);
    for (size_t i = 0; i < layers_.size(); ++i) {
        const auto &c = layers_[i];
        c.forward(x, gates, batch_input,
                  w.convolution.view(rows, c.geometry().kernel * 2 * hidden, Type::F32));
        ops::gated( gates, activated, ops::Activation::Tanh, time,
                   condition_sources ? TensorView{} : condition.columns(int(i) * 2 * hidden, 2 * hidden),
                   {}, packed, condition_sources, int(i) * 2 * hidden);
        bool last = i + 1 == layers_.size();
        auto projected = w.projected.view(rows, residuals_[i].output_channels(), type);
        residuals_[i].forward(activated, projected);
        ops::residual_split( projected, x, skip, i == 0, last, time,
                            mask.valid_rows_per_sequence, packed);
    }
    mask.output_rows = output_rows;
    post_.forward(skip, output, mask);
}
} // namespace tts
