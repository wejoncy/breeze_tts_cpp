#pragma once
#include "layers/weights.h"
#include "ops/convolution.h"
#include "core/forward_batch.h"

namespace tts {
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
} // namespace tts
