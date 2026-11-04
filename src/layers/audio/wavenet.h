#pragma once
#include "layers/convolution.h"
#include "layers/linear.h"
namespace tts {
class WaveNet {
    Linear pre_, conditioning_, post_;
    std::vector<Convolution> layers_;
    std::vector<Linear> residuals_;

  public:
    struct Workspace {
        Buffer hidden, skip, gates, activated, projected, condition, convolution;
        int max_batch, max_time, max_rows;
        Workspace(const WaveNet &, int batch, int time, BufferAllocator *, int sequences = 0);
    };
    WaveNet(Weights &, const std::string &, int layers, int dilation_rate);
    int condition_channels() const { return conditioning_.output_channels(); }
    void prepare(Workspace &, int rows) const;
    void forward(Workspace &, TensorView input, TensorView style, TensorView output,
                 const ForwardBatch &, int valid_time = 0,
                 const void *const *condition_sources = nullptr, void *const *output_rows = nullptr) const;
};
} // namespace tts
