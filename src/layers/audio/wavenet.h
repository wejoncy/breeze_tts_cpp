#pragma once
#include "core/buffer.h"
#include "layers/convolution.h"
#include "layers/linear.h"
namespace tts {
class WaveNet {
    Linear pre_, conditioning_, post_;
    std::vector<Convolution> layers_;
    std::vector<Linear> residuals_;

    struct Workspace {
        Buffer hidden, skip, gates, activated, projected, convolution;
        int max_batch, max_time, max_rows;
        Workspace(const WaveNet &, const ForwardContext &);
    };
  public:
    WaveNet(Weights &, const std::string &, int layers, int dilation_rate);
    int condition_channels() const { return conditioning_.output_channels(); }
    TensorView condition(TensorView style, ForwardContext &) const;
    void forward(TensorView input, TensorView condition, TensorView output,
                 ForwardContext &, TensorView output_rows = {}) const;

  private:
    void prepare(Workspace &, int rows) const;
    void forward(Workspace &, TensorView input, TensorView condition, TensorView output,
                 const ForwardBatch &, void *const *output_rows) const;
};
} // namespace tts
