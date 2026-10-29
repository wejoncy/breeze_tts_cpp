#pragma once
#include "request.h"
#include "graph.h"
#include <tuple>
namespace tts {
struct Model;
class Backend;
template <class Key, class GraphState> class ExecutionResources;
struct GenerationState;
class ModelRunner {
    struct Graphs {
        Graph first_depth, decoder, codec;
    };
    using GraphKey = std::tuple<int, int, int, float, float, int, float, float, bool, int, float,
                                float, bool, float, float, uint64_t>;

  public:
    using Resources = ExecutionResources<GraphKey, Graphs>;

  private:
    const Model &model;
    Backend &gpu;
    Resources &resources;
    int max_batch, context;
    void depth_frame(GenerationState &, int batch, int branches, const GenerationOptions &,
                     bool debug = false);
    void next_frame(GenerationState &, int batch, int branches, const GenerationOptions &);

  public:
    ModelRunner(const Model &m, Backend &g, Resources &r, int batch, int ctx)
        : model(m), gpu(g), resources(r), max_batch(batch), context(ctx) {}
    int generate(const std::vector<Request> &, const GenerationOptions &, AudioCallback, Stats &);
    std::vector<int> encode_audio(const std::vector<float> &);
    std::vector<float> decode_codes(const std::vector<int> &);
};
} // namespace tts
