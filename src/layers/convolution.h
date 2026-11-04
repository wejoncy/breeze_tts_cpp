#pragma once
#include "io/weights.h"
#include "ops/convolution.h"
#include "core/forward_batch.h"
#include "ops/audio.h"

namespace tts {
// Load-time checkpoint layout -> [group, phase, tap, output, input].
TensorView load_convolution_weight(Weights &, const std::string &name, bool transpose = false,
                                   int groups = 1, int phases = 1);
// All activations are [batch * time, channels]. Weights are explicitly packed
// once at load; execution projects directly from strided channel groups.
class Convolution {
    ops::ConvGeometry geometry_;
    TensorView weight_, bias_;

  public:
    Convolution(Weights &, const std::string &, ops::ConvGeometry);
    Convolution(Weights &, const std::string &, ops::ConvGeometry, TensorView bias);
    const ops::ConvGeometry &geometry() const {
        return geometry_;
    }
    Type type() const {
        return weight_.type;
    }
    size_t workspace_bytes(int batch, int input_length) const;
    void prepare(TensorView input, TensorView projection, int max_rows) const;
    void forward(TensorView x, TensorView y, const ForwardBatch &, TensorView workspace,
                 ops::Epilogue = {}, const SequenceLayout *output = nullptr) const;
};

// Causal convolution weights are shared by whole-sequence and history-backed
// forwards. Tensor inputs and borrowed histories select the execution form.
class CausalConvolution {
  public:
    struct Config {
        int stride = 1, dilation = 1;
        bool transpose = false, depthwise = false, replicate = false;
    };
    TensorView weight, bias;
    int in, out, kernel, stride, dilation;
    bool transpose, depthwise, replicate;
    CausalConvolution(Weights &, const std::string &, Config);
    CausalConvolution(Weights &, const std::string &, Config, TensorView bias);
    CausalConvolution(Weights &weights, const std::string &name)
        : CausalConvolution(weights, name, Config{}) {}
    int rows(int input) const {
        const int64_t result = transpose ? int64_t(input) * stride
                                        : input / stride + (input % stride != 0);
        if (input < 1 || result > INT32_MAX) throw std::invalid_argument("causal convolution row count");
        return int(result);
    }
    int history() const { return transpose ? (kernel - 1) / stride : (kernel - 1) * dilation; }
    size_t history_bytes(int batch, int time) const {
        return depthwise ? size_t(batch) * history() * in * sizeof(float)
                         : size_t(batch) * (time * (transpose ? stride : 1) +
                             (transpose ? kernel - stride : history())) * out * sizeof(float);
    }
    void prepare(TensorView input, TensorView projection, int max_input_rows) const;
    void forward(TensorView input, TensorView output, TensorView projection,
                  bool elu = false, TensorView residual = {}) const;
    void forward(TensorView input, TensorView output, TensorView projection,
                  const ForwardBatch &, const ConvHistory *) const;
  private:
    void project(TensorView input, TensorView projection, int max_input_rows = 0) const;
};
} // namespace tts
