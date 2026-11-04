#pragma once
#include "core/buffer.h"
#include "layers/convolution.h"
#include "core/forward_context.h"
#include "core/tensor_batch.h"
#include "io/json.h"

namespace tts {
class Vocoder {
    struct ResidualBlock {
        std::vector<Convolution> first, second;
    };
    struct Stage {
        Convolution upsample;
        std::vector<ResidualBlock> residuals;
    };
    Convolution pre_, post_;
    std::vector<Stage> stages_;
    std::pair<int64_t, int64_t> input_interval(int64_t begin, int64_t end) const;

    struct Workspace {
        Buffer activations[6], projection;
        int max_batch, max_frames;
        Workspace(const Vocoder &, int batch, int frames);
    };
    void prepare(Workspace &, int max_batch, int max_frames) const;
    void forward(Workspace &, TensorView mel, TensorView audio, const ForwardBatch &) const;
    TensorView forward(TensorView mel, TensorView destination, ForwardContext &) const;

  public:
    Vocoder(Weights &, const std::string &prefix, const Json &);
    int samples(int frames) const;
    // Structural dependency bounds, including residual branches and every
    // transposed convolution's stride phase. Returned frames are half-open.
    std::pair<int, int> input_frames(int first, int count, int total) const;
    int tile_capacity(int core_frames) const;
    TensorView forward(const TensorBatch &, TensorView layout, ForwardContext &) const;
};
} // namespace tts
