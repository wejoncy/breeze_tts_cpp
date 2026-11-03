#include "models/gpt_sovits/model_runner_internal.h"

namespace tts::gpt_sovits {
using namespace runner_detail;
namespace {
struct ZeroAllocator final : BufferAllocator {
    BufferAllocator &arena; Backend &gpu;
    ZeroAllocator(BufferAllocator &a, Backend &g) : arena(a), gpu(g) {}
    void *allocate(size_t bytes) override {
        auto *p = arena.allocate(bytes);
        if (bytes) CUDA(cudaMemsetAsync(p, 0, bytes, gpu.stream));
        return p;
    }
    void deallocate(void *p) noexcept override { arena.deallocate(p); }
    int device_ordinal() const override { return arena.device_ordinal(); }
};
} // namespace

size_t Runner::Impl::workspace_bytes(const Model &m, int batch, int context, int flow_time,
                              int vocoder_time, int max_flow_time, int max_vocoder_time) {
    size_t maximum = 0;
    auto measure = [&](auto construct) {
        WorkspacePlan plan;
        construct(&plan);
        maximum = std::max(maximum, plan.size);
    };
    measure([&](auto *p) {
        Bert::Workspace bert(m.bert, std::max(1, batch * context / 3),
                             m.config.at("bert").at("max_position_embeddings"), p,
                             batch * context);
        Semantic::Workspace semantic(m.semantic, batch, context, p);
    });
    measure([&](auto *p) { Semantic::Workspace w(m.semantic, batch, context, p); });
    measure([&](auto *p) {
        Pronunciation::Workspace w(m.pronunciation, m.pronunciation.max_tokens(), p);
    });
    measure([&](auto *p) { Acoustic::Workspace w(m.acoustic, 1, context, context, p, m.acoustic.feature_capacity(context)); });
    measure([&](auto *p) { ReferenceWork w(m, context, p); });
    // Synthesis phases borrow the arena in order; their outputs have
    // independent initialized backing and do not add to scratch capacity.
    measure([&](auto *p) { Flow::Workspace w(m.flow, batch, flow_time, p); });
    measure([&](auto *p) { Vocoder::Workspace w(m.vocoder, batch, vocoder_time, p); });
    measure([&](auto *p) { Flow::Workspace w(m.flow, 1, max_flow_time, p); });
    measure([&](auto *p) { Vocoder::Workspace w(m.vocoder, 1, max_vocoder_time, p); });
    return maximum;
}

size_t Runner::Impl::transfer_bytes() const {
    WorkspacePlan prefill, window, pronunciation, reference, sizing;
    prefill.allocate(metadata.capacity());
    prefill.allocate(AcousticMetadata::transfer_bytes(m.acoustic, capacity, context, true));
    for (const auto &s : storage)
        for (auto *b : {&s->table, &s->rng, &s->seen, &s->phones, &s->style,
                        &s->reference_mel, &s->reference_features, &s->reference_bert, &s->waveform_condition})
            prefill.allocate(b->size());
    for (auto *b : {&samplings, &generated, &ended}) prefill.allocate(b->size());
    window.allocate(metadata.capacity());
    window.allocate(AcousticMetadata::transfer_bytes(m.acoustic, capacity, context, false));
    for (auto *b : {&window_prefixes, &flow_prompt_rows, &flow_history_rows, &flow_feature_rows,
                    &flow_value_rows, &pcm})
        window.allocate(b->size());
    for (auto *b : {&pronunciation_ids, &pronunciation_positions, &pronunciation_characters,
                    &pronunciation_token_positions, &pronunciation_cu_seqlens,
                    &pronunciation_labels})
        pronunciation.allocate(b->size());
    ReferenceWork work(m, context, &sizing);
    for (auto *b : {&work.wave, &work.small, &work.phones}) reference.allocate(b->size());
    const auto &s = *storage.front();
    reference.allocate(std::max({work.encoder.codes.size(), s.style.size(),
                                 s.reference_mel.size(), s.reference_features.size(), s.waveform_condition.size()}));
    return std::max({prefill.size, window.size, pronunciation.size, reference.size});
}

Runner::Impl::Impl(const Model &model, Backend &backend, int batch, int length, int page, size_t runtime)
    : m(model), gpu(backend), capacity(batch), context(length),
      bert_time(m.config.at("bert").at("max_position_embeddings")),
      vocab(m.config.at("semantic").at("vocab_size")), eos(m.config.at("semantic").at("EOS")),
      core(m.config.at("synthesis").at("stream_core_frames")),
      future(m.config.at("synthesis").at("stream_future_frames")),
      holdback(m.config.at("synthesis").at("stream_holdback_frames")),
      left(m.config.at("synthesis").at("vocoder_context_frames")),
      fade(m.config.at("synthesis").at("stream_overlap_frames")),
      eos_min(m.config.at("synthesis").at("min_semantic_tokens")),
      flow_time(m.reference_frames + core + future), vocoder_time(left + core + future),
      max_flow_time(std::min(m.window_frames, m.reference_frames + m.target_frames)),
      max_vocoder_time(std::max(
          std::max(left, int(std::ceil(SynthesisWindows::maximum_overlap_seconds *
                  m.config.at("sample_rate").get<int>() / m.config.at("samples_per_frame").get<int>()))) +
              m.target_frames, m.vocoder.tile_capacity(m.target_frames))),
      type(m.semantic.type()),
      kv(gpu.memory, m.semantic.layers(), m.semantic.hidden(), type, batch, context, page),
      admissions(gpu.memory, kv, batch), slots(batch), resources(gpu.memory), shapes(batch), decode_plan(std::max(1, gpu.flash_decode_capacity(
          m.semantic.hidden() / m.semantic.heads(), m.semantic.type())), context),
      pronunciation_shapes(m.pronunciation.max_tokens()),
      bert_prefill_shapes(batch * context),
      semantic_prefill_shapes(batch * context),
      metadata(gpu.memory,
               Metadata::capacity(batch, context, bert_time, kv.block_stride(), max_flow_time)),
      logits(size_t(batch) * vocab * bytes(type), &gpu.memory),
      samplings(size_t(batch) * sizeof(SamplingRow), &gpu.memory),
      code_rows(size_t(batch) * sizeof(int *), &gpu.memory),
      caches(size_t(batch) * sizeof(CacheSequence), &gpu.memory),
      audio_positions(size_t(batch) * sizeof(int), &gpu.memory),
      generated(audio_positions.size(), &gpu.memory),
      ended(audio_positions.size(), &gpu.memory),
      pcm(size_t(batch) * m.vocoder.samples(max_vocoder_time) * sizeof(int16_t), &gpu.memory),
      mel(size_t(batch) * max_vocoder_time * m.mel_channels * bytes(type), &gpu.memory),
      audio(size_t(batch) * m.vocoder.samples(max_vocoder_time) * bytes(type), &gpu.memory),
      window_prefixes(size_t(batch) * sizeof(int), &gpu.memory),
      flow_prompt_rows(size_t(batch) * max_flow_time * sizeof(void *), &gpu.memory),
      flow_history_rows(flow_prompt_rows.size(), &gpu.memory),
      flow_feature_rows(flow_prompt_rows.size(), &gpu.memory),
      flow_value_rows(flow_prompt_rows.size(), &gpu.memory),
      pronunciation_ids(size_t(m.pronunciation.max_tokens()) * sizeof(int), &gpu.memory),
      pronunciation_positions(pronunciation_ids.size(), &gpu.memory),
      pronunciation_characters(pronunciation_ids.size(), &gpu.memory),
      pronunciation_labels(pronunciation_ids.size(), &gpu.memory),
      pronunciation_cu_seqlens((size_t(m.pronunciation.max_tokens()) + 1) * sizeof(int),
                             &gpu.memory),
      pronunciation_token_positions(pronunciation_ids.size(), &gpu.memory), host_ended(batch),
      host_pcm(pcm.size() / sizeof(int16_t)),
      host(batch, context, kv.block_stride(), max_flow_time,
           m.pronunciation.max_tokens(), eos) {
    ExecutionScope execution(gpu);
    auto &graphs = resources.graphs();
    if (core < 1 || future < 0 || holdback < 0 || core + future > m.target_frames ||
        core + future + m.reference_frames > m.window_frames || left < 0 ||
        left > m.rolling_frames || fade < 0 || fade > left || fade >= core || eos_min < 0)
        throw std::invalid_argument("invalid GPT-SoVITS streaming configuration");
    for (int i = 0; i < batch; ++i)
        storage.push_back(
            std::make_unique<Persistent>(m, context, kv.block_stride(), &gpu.memory));
    kv.reserve(batch * kv.block_stride());
    gpu.reserve_flash_decode_workspace(m.semantic.hidden() / m.semantic.heads(), type);
    workspace_capacity = workspace_bytes(m, batch, context, flow_time, vocoder_time,
                                         max_flow_time, max_vocoder_time);
    {
        WorkspacePlan minimum;
        Acoustic::Workspace work(m.acoustic, Acoustic::Capacity{capacity, context, context,
                                   m.acoustic.feature_capacity(context)}, &minimum);
        workspace_capacity = std::max(workspace_capacity, minimum.size);
    }
    resources.prepare(workspace_capacity);
    int lower = context;
    int upper = int(std::min<int64_t>(INT32_MAX, int64_t(capacity) * context));
    while (lower < upper) {
        int middle = lower + (upper - lower + 1) / 2;
        bool fits = false;
        try {
            WorkspacePlan plan;
            Acoustic::Workspace work(m.acoustic, Acoustic::Capacity{capacity, middle, middle,
                                       m.acoustic.feature_capacity(middle)}, &plan);
            fits = plan.size <= workspace_capacity;
        } catch (const std::overflow_error &) {}
        if (fits) lower = middle; else upper = middle - 1;
    }
    acoustic_capacity = {capacity, lower, lower, m.acoustic.feature_capacity(lower)};
    host.acoustic.reserve(m.acoustic, acoustic_capacity);
    metadata.reserve(std::max(metadata.capacity(), AcousticMetadata::bytes(m.acoustic, acoustic_capacity)));
    {
        auto lease = resources.lease_workspace();
        Acoustic::Workspace work(m.acoustic, acoustic_capacity, &resources.workspace());
        m.acoustic.prepare(work);
    }
    m.bert.prepare(batch * context);
    m.semantic.prepare(batch * context);
    m.pronunciation.prepare();
    m.reference.prepare();
    {
        const int64_t rows = int64_t(2) * batch * max_flow_time;
        if (rows > INT32_MAX)
            throw std::overflow_error("DiT planning row count");
        auto lease = resources.lease_workspace();
        const auto &estimator = m.flow.estimator();
        Flow::Workspace work(m.flow, 1, 1, &resources.workspace());
        estimator.prepare(work.estimator,
                          work.noise.view(1, m.mel_channels, estimator.type()),
                          work.velocity.view(1, m.mel_channels, estimator.type()), int(rows));
    }
    gpu.zero(samplings);
    gpu.zero(code_rows);
    gpu.zero(caches);
    gpu.zero(audio_positions);
    gpu.zero(generated);
    gpu.zero(ended);
    gpu.zero(flow_prompt_rows);
    gpu.zero(flow_history_rows);
    gpu.zero(flow_feature_rows);
    gpu.zero(flow_value_rows);
    gpu.zero(pronunciation_ids);
    gpu.zero(pronunciation_token_positions);
    gpu.zero(pronunciation_cu_seqlens);
    std::vector<int> inactive(m.pronunciation.max_tokens(), -1);
    gpu.upload(pronunciation_positions.data(), inactive.data(), pronunciation_positions.size());
    gpu.upload(pronunciation_characters.data(), inactive.data(), pronunciation_characters.size());
    std::vector<int> initial_lengths(size_t(m.pronunciation.max_tokens()) + 1, 1);
    initial_lengths[0] = 0;
    gpu.upload(pronunciation_cu_seqlens.data(), initial_lengths.data(), pronunciation_cu_seqlens.size());
    {
        auto lease = resources.lease_workspace();
        Pronunciation::Workspace work(m.pronunciation, m.pronunciation.max_tokens(),
                                      &resources.workspace());
        const int tokens = m.pronunciation.max_tokens();
        for (int rows : pronunciation_shapes.rows())
            graphs.register_shape("pronunciation", {rows, 1});
        for (int sequences = std::min(2, tokens); sequences <= tokens;) {
            graphs.register_shape("pronunciation", {tokens, sequences});
            if (sequences == tokens) break;
            sequences = int(std::min<int64_t>(tokens, int64_t(sequences) * 2));
        }
        // One execution path; sparse capacity pairs retain short-input work
        // without capturing the Cartesian product of both shape axes.
        graphs.capture("pronunciation", [&](const auto &shape) {
            m.pronunciation.forward(work, pronunciation_ids.ptr<int>(),
                pronunciation_positions.ptr<int>(), pronunciation_characters.ptr<int>(),
                pronunciation_cu_seqlens.ptr<int>(), shape[0], pronunciation_labels.ptr<int>(),
                shape[1], pronunciation_token_positions.ptr<int>());
        });
    }
    {
        auto meta_lease = ArenaLease(metadata);
        Metadata upload{metadata, gpu};
        PrefillMetadata descriptors;
        descriptors.reserve(capacity, context, kv.block_stride());
        descriptors.bind(upload, capacity, context, bert_time, kv.block_stride(), 1, 1,
                         kv.view());
        auto lease = resources.lease_workspace();
        Bert::Workspace bert_work(m.bert, std::max(1, capacity * context / 3), bert_time,
                                  &resources.workspace(), capacity * context);
        Semantic::Workspace work(m.semantic, capacity, context, &resources.workspace());
        for (int rows : bert_prefill_shapes.rows())
            graphs.register_shape("bert-prefill", {rows});
        graphs.capture("bert-prefill", [&](const auto &shape) {
            descriptors.bert_context.sequence.rows = shape[0];
            descriptors.bert_context.sequence.max_length = std::min(bert_time, shape[0]);
            auto features = m.bert.forward(bert_work, descriptors.device_ids, descriptors.bert_context);
            m.semantic.encode_features(work, features);
        });
        // The conditioning projection uses the same address for every BERT
        // bucket. Semantic graphs therefore need only the packed row count.
        descriptors.input.encoded_features = work.bert_projection.view(
            capacity * context, m.semantic.hidden(), type);
        for (int rows : semantic_prefill_shapes.rows())
            graphs.register_shape("semantic-prefill", {rows});
        graphs.capture("semantic-prefill", [&](const auto &shape) {
            descriptors.input.context.sequence.rows = shape[0];
            descriptors.input.context.sequence.max_length = std::min(context, shape[0]);
            descriptors.input.phone_rows.input_rows = shape[0];
            descriptors.input.audio_rows.input_rows = shape[0];
            m.semantic.prefill(work, descriptors.input, logits.view(capacity, vocab, type));
            sample(capacity);
        });
    }
    {
        auto lease = resources.lease_workspace();
        Semantic::Workspace work(m.semantic, batch, context, &resources.workspace());
        for (int rows : shapes.rows())
            for (int partitions : decode_plan.graph_splits(rows, m.semantic.heads()))
                graphs.register_shape("decode", {rows, partitions});
        graphs.capture("decode", [&](const auto &shape) { decode(work, shape[0], shape[1]); });
    }
    {
        auto lease = resources.lease_workspace();
        Flow::Workspace work(m.flow, batch, flow_time, &resources.workspace());
        const auto &reference = m.config.at("reference");
        const int shortest =
            std::min(m.reference_frames, int(reference.at("min_seconds").get<float>() *
                                             reference.at("mel_sample_rate").get<int>() /
                                             reference.at("mel_hop").get<int>()));
        for (int rows : shapes.rows()) {
            for (int prefix : {m.reference_frames, shortest}) {
                const int time = prefix + core + future;
                for (int history : {0, left}) {
                    graphs.register_shape("flow", {rows, time, prefix, history});
                }
            }
        }
        graphs.capture("flow", [&](const auto &shape) {
            flow_body(work, shape[0], shape[1], shape[2], shape[3], m.flow.sampling());
        });
    }
    {
        auto lease = resources.lease_workspace();
        int rows = window_batch(batch, max_flow_time, true);
        Flow::Workspace work(m.flow, rows, max_flow_time, &resources.workspace());
        graphs.warmup([&] {
            flow_body(work, rows, max_flow_time, m.reference_frames, left,
                      m.flow.sampling(0, 1));
        });
    }
    {
        auto lease = resources.lease_workspace();
        Vocoder::Workspace work(m.vocoder, 1, max_vocoder_time, &resources.workspace());
        m.vocoder.prepare(work, batch, max_vocoder_time);
    }
    for (int frames : {max_vocoder_time, vocoder_time, core + future}) {
        const int limit = window_batch(batch, frames, false);
        for (int rows : shapes.rows())
            if (rows <= limit) graphs.register_shape("audio", {frames, rows});
    }
    graphs.capture("audio", [&](const auto &shape) {
        auto lease = resources.lease_workspace();
        Vocoder::Workspace work(m.vocoder, window_batch(batch, shape[0], false), shape[0],
                                &resources.workspace());
        vocode(work, shape[1], shape[0]);
    });
    {
        auto lease = resources.lease_workspace();
        ZeroAllocator zero(resources.workspace(), gpu);
        ReferenceWork work(m, context, &zero);
        m.reference.forward(work.encoder,
                            work.wave.view(int(work.wave.size()/sizeof(float)), 1, Type::F32),
                            work.small.view(int(work.small.size()/bytes(type)), 1, type));
        gpu.sync();
    }
    {
        auto lease = resources.lease_workspace();
        ZeroAllocator zero(resources.workspace(), gpu);
        Acoustic::Workspace work(m.acoustic, 1, context, context, &zero, m.acoustic.feature_capacity(context));
        auto &slot = *storage.front();
        gpu.zero(slot.codes); gpu.zero(slot.phones); gpu.zero(slot.style);
        double speed = m.acoustic.capacity_speed(context);
        m.acoustic.forward(work, slot.codes.ptr<int>(), slot.phones.ptr<int>(),
                           slot.style.view(1, m.config.at("acoustic").at("gin_channels"), type),
                           slot.features.view(m.acoustic.feature_frames(context, speed), m.feature_channels, type),
                           m.acoustic.describe(1, context, context, speed));
        gpu.sync();
    }
    // Load the indirect kernel variants before requests enter the engine.
    {
        auto &slot = *storage.front();
        gpu.zero(slot.waveform_condition);
        auto &meta = host.acoustic;
        meta.clear(); meta.phones.append(1, 1);
        meta.phone_ids.push_back(slot.phones.ptr<int>());
        meta.text_outputs.push_back(slot.text_condition.data());
        for (bool text_only : {true, false}) {
            if (!text_only) {
                int hidden = m.acoustic.hidden_frames(1), frames = m.acoustic.feature_frames(1);
                meta.hidden.append(hidden, hidden); meta.resized.append(hidden, hidden);
                meta.features.append(frames, frames);
                meta.code_ids.push_back(slot.codes.ptr<int>());
                meta.styles.push_back(slot.style.data()); meta.conditions.push_back(slot.waveform_condition.data());
                meta.keys.push_back(slot.text_condition.data());
                meta.values.push_back(slot.text_condition.ptr<uint16_t>() + m.acoustic.text_condition_channels() / 2);
                meta.cross_attention.push_back({0, 0, hidden, 1});
                for (int p = 0; p < frames; ++p)
                    meta.feature_outputs.push_back(slot.features.ptr<uint16_t>() + size_t(p) * m.feature_channels);
            }
            auto metadata_lease = ArenaLease(metadata);
            Metadata upload{metadata, gpu};
            auto packed = meta.bind(upload, text_only);
            auto workspace_lease = resources.lease_workspace();
            Acoustic::Workspace work(m.acoustic, acoustic_capacity, &resources.workspace());
            if (text_only)
                m.acoustic.prepare_text(work, nullptr,
                    work.key_value.view(1, m.acoustic.text_condition_channels(), type), packed);
            else
                m.acoustic.forward(work, nullptr, nullptr, {},
                    work.features.view(packed.features.rows, m.feature_channels, type), packed);
        }
        gpu.sync();
    }
    gpu.seal_plans();
    resources.seal();
    gpu.memory.trim();
    gpu.memory.reserve_transfers(transfer_bytes());
    gpu.memory.reserve_runtime(runtime);
}
} // namespace tts::gpt_sovits
