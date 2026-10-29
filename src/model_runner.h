#pragma once
#include "request.h"
namespace tts {
struct Model;
class Backend;
class ExecutionResources;
struct GenerationState;
class ModelRunner {
    const Model &model;
    Backend &gpu;
    ExecutionResources &resources;
    int max_batch, context;
    void depth_frame(GenerationState &, int batch, int branches, const GenerationOptions &,
                     bool debug = false);
    void next_frame(GenerationState &, int batch, int branches, const GenerationOptions &);

  public:
    ModelRunner(const Model &m, Backend &g, ExecutionResources &r, int batch, int ctx)
        : model(m), gpu(g), resources(r), max_batch(batch), context(ctx) {}
    int generate(const std::vector<Request> &, const GenerationOptions &, AudioCallback, Stats &);
    std::vector<int> encode_audio(const std::vector<float> &);
    std::vector<float> decode_codes(const std::vector<int> &);
};
} // namespace tts
