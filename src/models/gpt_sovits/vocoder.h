#pragma once
#include "layers/convolution.h"
#include "io/json.h"

namespace tts::gpt_sovits {
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

  public:
    struct Workspace {
        Buffer activations[6], projection;
        int max_batch, max_frames;
        Workspace(const Vocoder &, int batch, int frames, BufferAllocator *);
    };
    Vocoder(Weights &, const Json &);
    int samples(int frames) const;
    // Structural dependency bounds, including residual branches and every
    // transposed convolution's stride phase. Returned frames are half-open.
    std::pair<int, int> input_frames(int first, int count, int total) const;
    int tile_capacity(int core_frames) const;
    void prepare(Workspace &, int max_batch, int max_frames) const;
    void forward(Workspace &, TensorView mel, TensorView audio, const ForwardBatch &) const;
};
} // namespace tts::gpt_sovits
