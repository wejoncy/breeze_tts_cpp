#pragma once
#include "tts/request.h"
#include "runtime/graph.h"
#include "runtime/graph_shapes.h"
#include <tuple>
#include <set>
namespace tts {
struct Model;
class Backend;
class PagedKVCache;
template <class Key, class GraphState> class ExecutionResources;
struct GenerationState;
struct RequestPlan;
class ModelRunner {
    struct Graphs {
        Graph decoder, codec;
    };
    const Model &model;
    Backend &gpu;
    int max_batch, context;
    GraphShapes graph_shapes_;
    uint64_t request_sequence_ = 0;
    std::unique_ptr<PagedKVCache> kv_;
    struct Slot;
    std::vector<std::unique_ptr<Slot>> slots_;
    struct DecodeBatch;
    struct Batch;
    std::unique_ptr<Batch> batch_;
    struct PrefillWork;
    std::unique_ptr<PrefillWork> prefill_;
    std::multiset<size_t> prefill_credits_;
    std::multiset<int> kv_credits_;
    void initialize_graphs();
    void depth_frame(GenerationState &, int batch, const DecodeBatch &, bool debug = false);
    void next_frame(GenerationState &, int batch, const DecodeBatch &);

  public:
    ModelRunner(const Model &, Backend &, int batch, int context, int page_tokens,
                size_t runtime_memory_bytes = 0);
    ~ModelRunner();
    // Executor-owned memory credits for the whole request, across input waves.
    // Must outlive every RequestState prepared from it.
    class RequestAdmission {
        friend class ModelRunner;
        struct Impl;
        std::unique_ptr<Impl> impl_;
        explicit RequestAdmission(std::unique_ptr<Impl>);

      public:
        ~RequestAdmission();
    };
    // CPU progress plus a lease on engine-owned request buffers and KV pages.
    // It owns no executor, model workspace, graph cache or private KV pool.
    class RequestState {
        friend class ModelRunner;
        struct Impl;
        std::unique_ptr<Impl> impl_;
        explicit RequestState(std::unique_ptr<Impl>);

      public:
        ~RequestState();
        bool done() const;
        bool prefilled() const;
        Stats stats() const;
        std::exception_ptr error() const;
    };
    std::unique_ptr<RequestAdmission> admit(RequestPlan &);
    std::unique_ptr<RequestState> prepare(RequestPlan &, size_t lane, RequestAdmission &);
    // Advance a packed prefill checkpoint; returns the number of inputs advanced.
    // Fixed pool capacity can reduce the offered batch without rejecting its inputs.
    size_t prefill_step(const std::vector<RequestState *> &);
    void step(const std::vector<RequestState *> &, const std::vector<AudioCallback> &);
    std::vector<int> encode_audio(const std::vector<float> &);
    std::vector<float> decode_codes(const std::vector<int> &);
};
} // namespace tts
