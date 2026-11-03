#pragma once
#include "models/gpt_sovits/model_runner.h"
#include "models/gpt_sovits/model.h"
#include "models/gpt_sovits/request_planner.h"
#include "runtime/execution_resources.h"
#include "runtime/graph_shapes.h"
#include "runtime/request_admission.h"
#include "runtime/request_slots.h"
#include "ops/audio.h"
#include "ops/sampling.h"
#include <chrono>
#include <tuple>

#include "models/gpt_sovits/model_runner_metadata.h"

namespace tts::gpt_sovits {
namespace runner_detail {
using Clock = std::chrono::steady_clock;
inline double now_seconds() {
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}
inline double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
// All temporary network workspaces borrow the same stable execution arena.
// Only request histories, metadata and graph handles persist across a step.
struct ReferenceWork {
    ReferenceEncoder::Workspace encoder;
    Acoustic::Workspace acoustic;
    Buffer wave, small, phones, features;
    ReferenceWork(const Model &m, int context, BufferAllocator *storage)
        : encoder(m.reference, storage), acoustic(m.acoustic, 1, context, context, storage),
          wave(size_t(m.config.at("reference").at("max_seconds").get<float>() *
                      m.config.at("reference").at("sampling_rate").get<int>()) *
                   sizeof(float),
               storage),
          small(size_t(m.config.at("reference").at("max_seconds").get<float>() *
                           m.config.at("reference").at("hubert_sample_rate").get<int>() +
                       m.config.at("reference").at("semantic_silence_seconds").get<float>() *
                           m.config.at("reference").at("sampling_rate").get<int>()) *
                    bytes(m.semantic.type()),
                storage),
          phones(size_t(context) * sizeof(int), storage),
          features(size_t(m.acoustic.feature_frames(context)) * m.feature_channels *
                       bytes(m.semantic.type()),
                   storage) {}
};
struct Persistent {
    Buffer table, codes, phones, seen, rng, done, style, reference_mel, reference_features,
        features, text_condition, reference_bert, waveform_condition;
    Buffer history, feature_history, offline_mel;
    Persistent(const Model &m, int context, int blocks, BufferAllocator *storage)
        : table(size_t(blocks) * sizeof(int), storage),
          codes(size_t(context) * sizeof(int), storage), phones(codes.size(), storage),
          seen(size_t(m.config.at("semantic").at("EOS").get<int>()) * sizeof(int), storage),
          rng(sizeof(uint64_t), storage), done(sizeof(int), storage),
          waveform_condition(size_t(m.acoustic.waveform_condition_channels()) * bytes(m.semantic.type()), storage),
          style(size_t(m.config.at("acoustic").at("gin_channels").get<int>()) *
                    bytes(m.semantic.type()),
                storage),
          reference_mel(size_t(m.reference_frames) * m.mel_channels * bytes(m.semantic.type()),
                        storage),
          reference_features(
              size_t(m.reference_frames) * m.feature_channels * bytes(m.semantic.type()), storage),
          features(size_t(m.acoustic.feature_capacity(context)) * m.feature_channels *
                       bytes(m.semantic.type()),
                   storage),
          text_condition(size_t(context) * m.acoustic.text_condition_channels() *
                             bytes(m.semantic.type()),
                         storage),
          reference_bert(size_t(context) *
                             m.semantic.hidden() * bytes(m.semantic.type()), storage) {
        size_t frames = size_t(m.acoustic.feature_capacity(context));
        history = Buffer(frames * m.mel_channels * bytes(m.semantic.type()), storage);
        offline_mel = Buffer(history.size(), storage);
        feature_history = Buffer(frames * m.feature_channels * bytes(m.semantic.type()), storage);
    }
};
} // namespace runner_detail
struct Runner::Impl {
    const Model &m;
    Backend &gpu;
    int capacity, context, bert_time, vocab, eos, core, future, holdback, left, fade, eos_min;
    int flow_time, vocoder_time, max_flow_time, max_vocoder_time;
    size_t workspace_capacity = 0;
    Acoustic::Capacity acoustic_capacity{};
    Type type;
    PagedKVCache kv;
    RequestAdmissionPool admissions;
    RequestSlots slots;
    std::vector<std::unique_ptr<runner_detail::Persistent>> storage;
    ExecutionResources resources;
    GraphShapes shapes;
    FlashDecodePlan decode_plan;
    GraphShapes pronunciation_shapes, bert_prefill_shapes, semantic_prefill_shapes;
    DeviceArena metadata;
    Buffer logits, samplings, code_rows, caches, audio_positions, generated, ended, pcm, mel, audio,
        window_prefixes, flow_prompt_rows, flow_history_rows, flow_feature_rows, flow_value_rows;
    Buffer pronunciation_ids, pronunciation_positions, pronunciation_characters,
        pronunciation_labels, pronunciation_cu_seqlens, pronunciation_token_positions;
    std::vector<int> host_ended;
    std::vector<int16_t> host_pcm;
    struct Admission final : RequestAdmission {
        Impl &owner;
        std::unique_ptr<RequestAdmissionPool::Lease> lease;
        Admission(Impl &r, std::unique_ptr<RequestAdmissionPool::Lease> l)
            : owner(r), lease(std::move(l)) {}
    };
    struct State final : RequestState {
        Impl &owner;
        std::unique_ptr<RequestAdmissionPool::Lease::RowLease> row_credit;
        std::unique_ptr<RequestSlots::Lease> slot;
        runner_detail::Persistent &data;
        PlannedInput input;
        GenerationOptions options;
        std::unique_ptr<PagedKVCache::Lease> cache;
        runner_detail::Clock::time_point start = runner_detail::Clock::now();
        runner_detail::Clock::time_point segment_started = start;
        Stats metrics;
        std::vector<int16_t> pending;
        int count = 0, cursor = 0, windows = 0, history_count = 0, prompt = 0, feature_codes = 0, pcm_cursor = 0;
        int segment_start_frames = 0;
        size_t segment = 0;
        uint64_t seed;
        WindowSchedule schedule;
        bool is_prefilled = false, semantic_done = false, finished = false;
        State(Impl &r, std::unique_ptr<RequestAdmissionPool::Lease::RowLease> credit,
              std::unique_ptr<RequestSlots::Lease> lease, PlannedInput p, GenerationOptions opt)
            : owner(r), row_credit(std::move(credit)), slot(std::move(lease)), data(*r.storage[slot->index]), input(std::move(p)),
              options(opt), seed(opt.seed) {
            pending.reserve(r.m.vocoder.samples(input.windows.fade));
            prompt = int(input.text.phones.size() + input.reference->text.phones.size() +
                         input.reference->codes.size());
        }
        bool done() const override {
            return finished;
        }
        bool prefilled() const override {
            return is_prefilled;
        }
        Stats stats() const override {
            return metrics;
        }
        std::exception_ptr error() const override {
            return {};
        }
    };
    using WindowShape = std::tuple<int, int, int, int, int, float, SynthesisMode>;
    // One execution thread reuses these descriptors. Outer membership, nested
    // sampling, and inner flow chunks have separate storage while live.
    struct HostScratch {
        std::vector<State *> rows, decode;
        std::vector<SamplingRow> samples;
        std::vector<CacheSequence> sequences;
        std::vector<const int *> tokens;
        std::vector<int> counts, positions, cache_positions, prefixes, seen, lengths;
        std::vector<const void *> feature_rows, prompt_rows;
        std::vector<void *> history_rows, feature_output_rows, value_rows;
        std::vector<int> mel_rows;
        std::vector<uint64_t> seeds, windows;
        DiT::Condition condition;
        std::vector<std::pair<WindowShape, size_t>> groups;
        std::vector<size_t> members, chunk;
        runner_detail::PrefillMetadata prefill;
        runner_detail::AcousticMetadata acoustic;
        std::vector<int> pronunciation_positions, pronunciation_characters;
        SequenceBuilder pronunciation_rows;
        HostScratch(int batch, int context, int blocks, int time, int pronunciation, int eos)
            : seen(eos), lengths(1) {
            for (auto *v : {&rows, &decode}) v->reserve(batch);
            samples.reserve(batch);
            sequences.reserve(batch);
            tokens.reserve(batch);
            for (auto *v : {&counts, &positions, &cache_positions, &prefixes}) v->reserve(batch);
            for (auto *v : {&feature_rows, &prompt_rows}) v->reserve(size_t(batch) * time);
            for (auto *v : {&history_rows, &feature_output_rows, &value_rows})
                v->reserve(size_t(batch) * time);
            mel_rows.reserve(size_t(batch) * time);
            for (auto *v : {&seeds, &windows}) v->reserve(batch);
            condition.mel.reserve(size_t(batch) * 2);
            groups.reserve(batch);
            for (auto *v : {&members, &chunk}) v->reserve(batch);
            prefill.reserve(batch, context, blocks);
            for (auto *v : {&pronunciation_positions, &pronunciation_characters}) v->reserve(pronunciation);
            pronunciation_rows.reserve(pronunciation, pronunciation);
        }
    } host;
    State &state(RequestState &base);
    static size_t workspace_bytes(const Model &m, int batch, int context, int flow_time,
                                  int vocoder_time, int max_flow_time, int max_vocoder_time);
    size_t transfer_bytes() const;
    Impl(const Model &model, Backend &backend, int batch, int length, int page, size_t runtime);
    void sample(int rows);
    void decode(Semantic::Workspace &work, int rows, int partitions = 1);
    int window_batch(int requested, int frames, bool flow) const;
    void flow_body(Flow::Workspace &work, int batch, int time, int prefix, int history,
                   FlowSampling parameters, bool offline = false, int first = 0);
    void vocode(Vocoder::Workspace &work, int batch, int frames, int first = 0);
    void sampling(const std::vector<State *> &rows, int padded);
    void sampled(const std::vector<State *> &rows);
    void prefill(const std::vector<State *> &rows);
    void acoustic_batch(const std::vector<State *> &, const std::vector<size_t> *members, bool text_only);
    void advance(const std::vector<State *> &rows);
    int total_frames(const State &s) const;
    void prepare_segment(State &);
    int ready_frames(const State &s) const;
    void synthesize(const std::vector<State *> &, const std::vector<AudioCallback> &);
    int flow_window(const std::vector<State *> &, const WindowShape &,
                    const std::vector<size_t> &, int first);
    void synthesize_window(const std::vector<State *> &, const std::vector<AudioCallback> &,
                           const WindowShape &, const std::vector<size_t> &);
    void emit_window(State &, const AudioCallback &, int16_t *, int accepted, int history);
    void synthesize_offline(State &, const AudioCallback &);
    void emit(State &s, const AudioCallback &callback, const int16_t *audio, size_t count);
    void emit(State &s, const AudioCallback &callback, const std::vector<int16_t> &audio);
};
} // namespace tts::gpt_sovits
