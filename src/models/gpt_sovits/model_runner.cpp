#include "models/gpt_sovits/model_runner_internal.h"

namespace tts::gpt_sovits {
using namespace runner_detail;

Runner::Impl::State &Runner::Impl::state(RequestState &base) {
    auto &s = dynamic_cast<State &>(base);
    if (&s.owner != this)
        throw std::invalid_argument("request belongs to another model runner");
    return s;
}

void Runner::Impl::sample(int rows) {
    tts::sample(gpu.stream, logits.view(rows, vocab, type), {1, eos, eos},
                samplings.ptr<SamplingRow>(), rows, 0, ended.ptr<int>());
}

void Runner::Impl::decode(Semantic::Workspace &work, int rows, int partitions) {
    ForwardBatch input{rows, 1};
    input.cache = kv.view();
    input.cache.layout.sequences = caches.ptr<CacheSequence>();
    input.capacity = context;
    input.decode = true;
    input.decode_partitions = partitions;
    m.semantic.decode(work, nullptr, audio_positions.ptr<int>(), input,
                      logits.view(rows, vocab, type), code_rows.ptr<const int *>());
    sample(rows);
}

int Runner::Impl::window_batch(int requested, int frames, bool flow) const {
    int lower = 0, upper = requested;
    while (lower < upper) {
        int rows = lower + (upper - lower + 1) / 2;
        WorkspacePlan plan;
        if (flow) { Flow::Workspace work(m.flow, rows, frames, &plan); }
        else { Vocoder::Workspace work(m.vocoder, rows, frames, &plan); }
        if (plan.size <= workspace_capacity) lower = rows;
        else upper = rows - 1;
    }
    if (!lower) throw std::logic_error("window exceeds initialized execution workspace");
    return lower;
}

void Runner::Impl::flow_body(Flow::Workspace &work, int batch, int time, int prefix, int history,
               FlowSampling parameters, bool offline, int first) {
    ForwardBatch input{batch, time};
    for (int step = 0; step + 1 < parameters.steps; ++step)
        m.flow_step(work, input, window_prefixes.ptr<int>(), prefix, step, parameters);
    int frames = history + time - prefix;
    ops::EulerOutput output{offline ? TensorView{}
                               : mel.view((first + batch) * frames, m.mel_channels, type)
                                     .slice(first * frames, batch * frames),
                            prefix - history, frames, m.mel_minimum, m.mel_maximum,
                            flow_prompt_rows.ptr<const void *>(), flow_history_rows.ptr<void *>(),
                            offline ? flow_value_rows.ptr<void *>() : nullptr};
    m.flow_step(work, input, window_prefixes.ptr<int>(), prefix,
                parameters.steps - 1, parameters, output);
}

void Runner::Impl::vocode(Vocoder::Workspace &work, int batch, int frames, int first) {
    auto output = audio.view(batch * m.vocoder.samples(frames), 1, type);
    m.vocoder.forward(work,
                      mel.view((first + batch) * frames, m.mel_channels, type)
                          .slice(first * frames, batch * frames),
                      output, ForwardBatch{batch, frames});
    tts::pcm(gpu.stream, output, pcm.ptr<int16_t>());
}

void Runner::Impl::sampling(const std::vector<State *> &rows, int padded) {
    auto &samples = host.samples;
    auto &counts = host.counts;
    samples.assign(padded, {});
    counts.assign(padded, 0);
    for (size_t i = 0; i < rows.size(); ++i) {
        auto &s = *rows[i];
        auto &r = samples[i];
        r.codes = s.data.codes.ptr<int>() + s.count;
        r.done = s.data.done.ptr<int>();
        r.seen = s.data.seen.ptr<int>();
        r.rng = s.data.rng.ptr<uint64_t>();
        r.logits[0] = int(i);
        r.penalty = s.options.repetition_penalty;
        r.generated = generated.ptr<int>() + i;
        r.eos_min_tokens = eos_min;
        r.eos_on_argmax = true;
        r.backbone = {s.options.do_sample ? s.options.temperature : 0.f, s.options.top_p,
                      s.options.top_k, true, false};
        counts[i] = s.count;
    }
    gpu.upload(samplings.data(), samples.data(), samples.size() * sizeof(SamplingRow));
    gpu.upload(generated.data(), counts.data(), counts.size() * sizeof(int));
}

void Runner::Impl::sampled(const std::vector<State *> &rows) {
    gpu.download(host_ended.data(), ended.data(), rows.size() * sizeof(int));
    for (size_t i = 0; i < rows.size(); ++i) {
        auto &s = *rows[i];
        if (host_ended[i])
            s.semantic_done = true;
        else if (++s.count == s.input.max_codes) {
            s.semantic_done = true;
            s.metrics.truncated = 1;
        }
    }
}

int Runner::Impl::total_frames(const State &s) const {
    return std::min(s.options.max_frames - s.segment_start_frames,
                    m.acoustic.feature_frames(s.count, s.options.speed_factor));
}

int Runner::Impl::ready_frames(const State &s) const {
    if (s.finished || !s.count) return 0;
    int accepted = s.input.windows.ready(total_frames(s), s.cursor, s.semantic_done, s.schedule,
                                         now_seconds());
    if (accepted && !s.semantic_done) {
        int needed = int(std::ceil((s.cursor + accepted + s.input.windows.lookahead(s.cursor) +
                                   s.input.windows.holdback) * s.options.speed_factor /
                                  m.acoustic.feature_frames(1)));
        if (s.count < needed) return 0;
    }
    return accepted;
}

void Runner::Impl::emit(State &s, const AudioCallback &callback, const int16_t *audio, size_t count) {
    if (!count)
        return;
    if (!s.metrics.frames)
        s.metrics.first_audio_ms = elapsed(s.start);
    s.metrics.frames += int(count / m.vocoder.samples(1));
    s.metrics.audio_seconds += double(count) / m.config.at("sample_rate").get<int>();
    if (!callback(0, audio, count))
        s.finished = true;
}

void Runner::Impl::emit(State &s, const AudioCallback &callback, const std::vector<int16_t> &audio) {
    emit(s, callback, audio.data(), audio.size());
}


Runner::Runner(const Model &m, Backend &gpu, int batch, int context, int page, size_t runtime)
    : impl_(std::make_unique<Impl>(m, gpu, batch, context, page, runtime)) {}
Runner::~Runner() = default;
size_t Runner::max_step_samples() const {
    const auto &r = *impl_;
    const int pause = int(std::llround(maximum_fragment_interval *
        r.m.config.at("sample_rate").get<int>() / r.m.config.at("samples_per_frame").get<int>()));
    // A step can emit one window, its pending tail, and one segment pause.
    return size_t(r.m.vocoder.samples(r.max_vocoder_time)) + r.m.vocoder.samples(pause);
}
void Runner::prepare_task(PreparationTask &base) {
    auto &r = *impl_;
    auto &task = dynamic_cast<text::G2PWTask &>(base);
    CUDA(cudaSetDevice(r.gpu.device));
    ExecutionScope execution(r.gpu);
    const int capacity = r.m.pronunciation.max_tokens();
    int tokens = int(task.ids.size()), queries = int(task.positions.size());
    if (tokens < 1 || tokens > capacity || queries < 1 || queries > tokens ||
        task.characters.size() != size_t(queries) || task.offsets.size() < 2 ||
        task.offsets.front() != 0 || task.offsets.back() != tokens ||
        std::any_of(task.positions.begin(), task.positions.end(),
                    [&](int pos) { return pos < 0 || pos >= tokens; }))
        throw std::invalid_argument("invalid pronunciation task dimensions");
    int batch = int(task.offsets.size()) - 1;
    auto &graphs = r.resources.graphs();
    const auto *selected = graphs.select("pronunciation", {tokens, batch});
    if (!selected)
        throw std::invalid_argument("pronunciation exceeds initialized graph capacities");
    const int padded = (*selected)[0], sequence_capacity = (*selected)[1];
    auto &positions = r.host.pronunciation_positions;
    auto &characters = r.host.pronunciation_characters;
    auto &sequences = r.host.pronunciation_rows;
    sequences.clear();
    positions.assign(padded, -1);
    characters.assign(padded, -1);
    for (int i = 0; i < batch; ++i) {
        int start = task.offsets[i], length = task.offsets[i + 1] - start;
        if (length < 1 || length > capacity)
            throw std::invalid_argument("invalid pronunciation sequence length");
        sequences.append(length, length);
    }
    std::copy(task.positions.begin(), task.positions.end(), positions.begin());
    std::copy(task.characters.begin(), task.characters.end(), characters.begin());
    r.gpu.upload(r.pronunciation_ids.data(), task.ids.data(), task.ids.size() * sizeof(int));
    r.gpu.upload(r.pronunciation_positions.data(), positions.data(), positions.size() * sizeof(int));
    r.gpu.upload(r.pronunciation_characters.data(), characters.data(), characters.size() * sizeof(int));
    const auto sequence_rows = sequences.padded_view(padded, sequence_capacity);
    r.gpu.upload(r.pronunciation_token_positions.data(), sequence_rows.row_position,
                 size_t(padded) * sizeof(int));
    r.gpu.upload(r.pronunciation_cu_seqlens.data(), sequence_rows.cu_seqlens,
                 (size_t(sequence_capacity) + 1) * sizeof(int));
    graphs.replay("pronunciation", {padded, sequence_capacity});
    task.labels.resize(queries);
    r.gpu.download(task.labels.data(), r.pronunciation_labels.data(), queries * sizeof(int));
}

std::unique_ptr<ModelRunner::RequestAdmission> Runner::admit(RequestPlan &base) {
    auto &r = *impl_;
    auto &plan = dynamic_cast<Plan &>(base);
    CUDA(cudaSetDevice(r.gpu.device));
    ExecutionScope execution(r.gpu);
    std::vector<int> pages;
    for (const auto &input : plan.inputs)
        pages.push_back(r.kv.required_pages({input.capacity}));
    auto lease = r.admissions.admit(pages, 0);
    if (!lease) return {};
    return std::make_unique<Impl::Admission>(r, std::move(lease));
}
std::unique_ptr<ModelRunner::RequestState> Runner::prepare(RequestPlan &base, size_t input,
                                                           RequestAdmission &admission) {
    auto &r = *impl_;
    auto &a = dynamic_cast<Impl::Admission &>(admission);
    auto &plan = dynamic_cast<Plan &>(base);
    if (&a.owner != &r)
        throw std::invalid_argument("admission belongs to another model runner");
    auto &planned = plan.inputs.at(input);
    auto row_credit = a.lease->acquire_row();
    if (!row_credit) return {};
    auto options = plan.options;
    options.seed = request_seed(options.seed, input);
    auto s = std::make_unique<Impl::State>(r, std::move(row_credit), r.slots.acquire(),
                                           std::move(planned),
                                           options);
    auto &d = s->data;
    const auto &ref = *s->input.reference;
    r.prepare_segment(*s);
    r.gpu.upload(d.style.data(), ref.style.data(), ref.style.size() * sizeof(uint16_t));
    if (ref.waveform_condition.size() * sizeof(uint16_t) != d.waveform_condition.size())
        throw std::invalid_argument("reference waveform conditioning dimensions mismatch");
    r.gpu.upload(d.waveform_condition.data(), ref.waveform_condition.data(), d.waveform_condition.size());
    r.gpu.upload(d.reference_mel.data(), ref.mel.data(), ref.mel.size() * sizeof(uint16_t));
    r.gpu.upload(d.reference_features.data(), ref.features.data(),
                 ref.features.size() * sizeof(uint16_t));
    if (ref.bert_projection.size() != ref.text.bert_ids.size() * r.m.semantic.hidden() ||
        ref.bert_projection.size() * sizeof(uint16_t) > d.reference_bert.size())
        throw std::invalid_argument("reference text conditioning dimensions mismatch");
    r.gpu.upload(d.reference_bert.data(), ref.bert_projection.data(),
                 ref.bert_projection.size() * sizeof(uint16_t));
    return s;
}

void Runner::Impl::prepare_segment(State &s) {
    s.prompt = int(s.input.text.phones.size() + s.input.reference->text.phones.size() +
                   s.input.reference->codes.size());
    s.cache.reset();
    s.cache = std::make_unique<PagedKVCache::Lease>(
        kv, std::vector<int>{s.prompt}, std::vector<int>{s.input.capacity}, s.data.table.ptr<int>());
    s.options.seed = request_seed(s.seed, s.segment);
    gpu.zero(s.data.done);
    gpu.upload(s.data.rng.data(), &s.options.seed, sizeof(uint64_t));
    auto &seen = host.seen;
    std::fill(seen.begin(), seen.end(), 0);
    for (int code : s.input.reference->codes) seen.at(code) = 1;
    gpu.upload(s.data.seen.data(), seen.data(), s.data.seen.size());
    gpu.upload(s.data.phones.data(), s.input.text.phones.data(), s.input.text.phones.size() * sizeof(int));
    s.count = s.cursor = s.windows = s.history_count = s.feature_codes = s.pcm_cursor = 0;
    s.schedule = WindowSchedule(s.input.windows.core, s.input.windows.maximum, now_seconds());
    s.is_prefilled = s.semantic_done = false;
    s.segment_start_frames = s.metrics.frames;
    if (s.segment) {
        s.segment_started = Clock::now();
        s.input.max_codes = m.acoustic.codes_for_frames(
            s.options.max_frames - s.metrics.frames, s.options.speed_factor, context - s.prompt);
    }
}

void Runner::Impl::acoustic_batch(const std::vector<State *> &rows,
                                    const std::vector<size_t> *selection, bool text_only) {
    auto &meta = host.acoustic;
    size_t cursor = 0, selected = selection ? selection->size() : rows.size();
    const int repeat = m.acoustic.hidden_frames(1);
    while (cursor < selected) {
        meta.clear();
        int code_total = 0, phone_total = 0, frame_total = 0;
        while (cursor < selected) {
            size_t index = selection ? (*selection)[cursor] : cursor;
            auto &s = *rows[index];
            if (!text_only && s.feature_codes == s.count) { ++cursor; continue; }
            int phones = int(s.input.text.phones.size());
            int codes = text_only ? 0 : s.count;
            int frames = text_only ? 0 : m.acoustic.feature_frames(codes, s.options.speed_factor);
            if (phones < 1 || phones > context || (!text_only && (codes < 1 || codes > context)) ||
                size_t(frames) * m.feature_channels * bytes(type) > s.data.features.size())
                throw std::invalid_argument("acoustic request sequence exceeds initialized storage");
            bool fits = int64_t(phone_total) + phones <= acoustic_capacity.phones &&
                        int64_t(code_total) + codes <= acoustic_capacity.codes &&
                        int64_t(frame_total) + frames <= acoustic_capacity.frames &&
                        meta.members.size() < size_t(acoustic_capacity.sequences);
            if (!fits) {
                if (meta.members.empty()) throw std::logic_error("acoustic sequence exceeds initialized total-row budget");
                break;
            }
            meta.phones.append(phones, phones);
            if (text_only) {
                for (int p = 0; p < phones; ++p) {
                    meta.phone_ids.push_back(s.data.phones.ptr<int>() + p);
                    meta.text_outputs.push_back(s.data.text_condition.ptr<uint16_t>() +
                                               size_t(p) * m.acoustic.text_condition_channels());
                }
            } else {
                int time = m.acoustic.hidden_frames(codes);
                int resized = m.acoustic.hidden_frames(codes, s.options.speed_factor);
                int valid = m.acoustic.valid_frames(codes, s.options.speed_factor);
                int offset = code_total * repeat;
                meta.hidden.append(time, time); meta.resized.append(resized, resized);
                meta.features.append(frames, valid);
                meta.cross_attention.push_back({offset, 0, time, phones});
                for (int p = 0; p < codes; ++p) meta.code_ids.push_back(s.data.codes.ptr<int>() + p);
                meta.styles.push_back(s.data.style.data());
                meta.conditions.push_back(s.data.waveform_condition.data());
                meta.keys.push_back(s.data.text_condition.data());
                meta.values.push_back(s.data.text_condition.ptr<uint16_t>() + m.acoustic.text_condition_channels() / 2);
                for (int p = 0; p < frames; ++p)
                    meta.feature_outputs.push_back(s.data.features.ptr<uint16_t>() + size_t(p) * m.feature_channels);
            }
            phone_total += phones; code_total += codes; frame_total += frames;
            meta.members.push_back(index); ++cursor;
        }
        if (meta.members.empty()) continue;
        Range range(text_only ? "acoustic text" : "acoustic features");
        auto metadata_lease = ArenaLease(metadata);
        Metadata upload{metadata, gpu};
        auto packed = meta.bind(upload, text_only);
        auto workspace_lease = resources.lease_workspace();
        Acoustic::Workspace work(m.acoustic, acoustic_capacity, &resources.workspace());
        if (text_only) {
            m.acoustic.prepare_text(work, nullptr,
                work.key_value.view(packed.phones.rows, m.acoustic.text_condition_channels(), type),
                packed);
        } else {
            m.acoustic.forward(work, nullptr, nullptr, {},
                work.features.view(packed.features.rows, m.feature_channels, type), packed);
            for (size_t index : meta.members) rows[index]->feature_codes = rows[index]->count;
        }
    }
}

void Runner::Impl::prefill(const std::vector<State *> &rows) {
    Range range("packed prefill");
    auto meta_lease = ArenaLease(metadata);
    Metadata upload{metadata, gpu};
    auto &descriptors = host.prefill;
    descriptors.clear();
    int max_bert = 0;
    for (size_t i = 0; i < rows.size(); ++i) {
        auto &s = *rows[i];
        const auto &ref = *s.input.reference;
        int offset = descriptors.semantic.rows(),
            phone = int(ref.text.phones.size() + s.input.text.phones.size());
        // The reference BERT projection is input-independent and was prepared
        // once with the reference. Only target text enters this BERT batch.
        for (int p : ref.text.phone_bert_rows) {
            descriptors.phone_bert.push_back(-1);
            descriptors.reference_rows.push_back(
                p < 0 ? nullptr : s.data.reference_bert.ptr<uint16_t>() + size_t(p) * m.semantic.hidden());
        }
        descriptors.phones.insert(descriptors.phones.end(), ref.text.phones.begin(),
                                  ref.text.phones.end());
        {
            const auto *text = &s.input.text;
            int base = int(descriptors.ids.size());
            for (int p : text->phone_bert_rows) {
                descriptors.phone_bert.push_back(p < 0 ? -1 : base + p);
                descriptors.reference_rows.push_back(nullptr);
            }
            max_bert = std::max(max_bert, descriptors.append_bert(*text));
            descriptors.phones.insert(descriptors.phones.end(), text->phones.begin(),
                                      text->phones.end());
        }
        for (int p = 0; p < phone; ++p) {
            descriptors.phone_rows.push_back(offset + p);
            descriptors.phone_positions.push_back(p);
        }
        descriptors.codes.insert(descriptors.codes.end(), ref.codes.begin(), ref.codes.end());
        for (int p = 0; p < int(ref.codes.size()); ++p) {
            descriptors.code_output.push_back(offset + phone + p);
            descriptors.code_positions.push_back(p);
        }
        descriptors.semantic.append(s.prompt, s.prompt);
        descriptors.last.push_back(descriptors.semantic.rows() - 1);
        descriptors.prefixes.push_back({offset, 0, phone, phone});
        descriptors.suffixes.push_back({offset + phone, 0, int(ref.codes.size()), s.prompt});
        const auto &table = s.cache->block_table();
        descriptors.tables.insert(descriptors.tables.end(), table.begin(), table.end());
    }
    const int actual_bert = int(descriptors.ids.size()),
              actual_rows = descriptors.semantic.rows();
    bool use_graph = std::all_of(rows.begin(), rows.end(),
                                 [](auto *s) { return s->options.use_cuda_graph; });
    const int bert_batch = descriptors.bert.size();
    int bert_rows = use_graph && actual_bert && bert_batch <= 2 * capacity
                        ? bert_prefill_shapes.select(actual_bert) : 0;
    int semantic_rows = use_graph ? semantic_prefill_shapes.select(actual_rows) : 0;
    descriptors.bind(upload, capacity, context, bert_time, kv.block_stride(),
                     bert_rows ? bert_rows : actual_bert,
                     semantic_rows ? semantic_rows : actual_rows, kv.view());
    sampling(rows, capacity);
    {
        auto lease = resources.lease_workspace();
        Bert::Workspace bert_work(m.bert, std::max(1, capacity * context / 3), bert_time,
                                  &resources.workspace(), capacity * context);
        Semantic::Workspace work(m.semantic, capacity, context, &resources.workspace());
        auto &graphs = resources.graphs();
        if (actual_bert)
            graphs.run("bert-prefill", {bert_rows}, bert_rows != 0, [&] {
                descriptors.bert_context.sequence.sequences = bert_batch;
                descriptors.bert_context.sequence.max_length = max_bert;
                auto features = m.bert.forward(bert_work, descriptors.device_ids,
                                               descriptors.bert_context);
                m.semantic.encode_features(work, features);
            });
        descriptors.input.encoded_features = work.bert_projection.view(
            capacity * context, m.semantic.hidden(), type);
        graphs.run("semantic-prefill", {semantic_rows}, semantic_rows != 0, [&] {
            m.semantic.prefill(work, descriptors.input, logits.view(capacity, vocab, type));
            sample(capacity);
        });
    }
    // Text conditioning is independent of the generated semantic tokens. Queue
    // it behind prefill now, so its host-side GEMM preparation overlaps the
    // already submitted networks instead of opening a gap at the first PCM.
    acoustic_batch(rows, nullptr, true);
    sampled(rows);
    for (auto *s : rows) {
        s->is_prefilled = true;
        s->metrics.prefill_ms += elapsed(s->segment_started);
    }
}

size_t Runner::prefill_step(const std::vector<RequestState *> &requests) {
    auto &r = *impl_;
    CUDA(cudaSetDevice(r.gpu.device));
    ExecutionScope execution(r.gpu);
    auto &rows = r.host.rows;
    rows.clear();
    for (auto *request : requests) {
        auto &s = r.state(*request);
        if (!s.prefilled() && !s.done())
            rows.push_back(&s);
    }
    if (!rows.empty())
        r.prefill(rows);
    return rows.size();
}

void Runner::Impl::advance(const std::vector<State *> &rows) {
    if (rows.empty())
        return;
    Range range("semantic decode");
    int padded = shapes.select(int(rows.size()));
    if (!padded)
        padded = int(rows.size());
    auto &sequences = host.sequences;
    auto &tokens = host.tokens;
    auto &positions = host.positions;
    auto &cache_positions = host.cache_positions;
    sequences.assign(padded, {});
    tokens.assign(padded, nullptr);
    positions.assign(padded, 0);
    cache_positions.assign(padded, 0);
    // The positions are host-owned descriptors, uploaded once for this batch.
    auto meta_lease = ArenaLease(metadata);
    Metadata upload{metadata, gpu};
    for (size_t i = 0; i < rows.size(); ++i) {
        auto &s = *rows[i];
        host.lengths[0] = s.prompt + s.count;
        s.cache->ensure(host.lengths);
        cache_positions[i] = s.prompt + s.count - 1;
        positions[i] = int(s.input.reference->codes.size()) + s.count - 1;
        tokens[i] = s.data.codes.ptr<int>() + s.count - 1;
        sequences[i].blocks = s.data.table.ptr<int>();
    }
    int *device_positions = upload.put(cache_positions);
    for (size_t i = 0; i < rows.size(); ++i)
        sequences[i].position = device_positions + i;
    gpu.upload(caches.data(), sequences.data(), sequences.size() * sizeof(CacheSequence));
    gpu.upload(code_rows.data(), tokens.data(), tokens.size() * sizeof(int *));
    gpu.upload(audio_positions.data(), positions.data(), positions.size() * sizeof(int));
    sampling(rows, padded);
    bool graph =
        std::all_of(rows.begin(), rows.end(), [](auto *s) { return s->options.use_cuda_graph; });
    int key_length = *std::max_element(cache_positions.begin(), cache_positions.end()) + 1;
    int partitions = decode_plan.select(padded, m.semantic.heads(), key_length);
    resources.graphs().run("decode", {padded, partitions}, graph, [&] {
        auto lease = resources.lease_workspace();
        Semantic::Workspace work(m.semantic, capacity, context, &resources.workspace());
        decode(work, padded, partitions);
    });
    sampled(rows);
}

void Runner::Impl::synthesize_offline(State &s, const AudioCallback &callback) {
    Range range("offline vocoder");
    const int total = total_frames(s), core_frames = m.target_frames;
    auto lease = resources.lease_workspace();
    auto mel = s.data.offline_mel.view(m.acoustic.feature_capacity(context), m.mel_channels, type);
    const int first = s.pcm_cursor, count = std::min(core_frames, total - first);
    if (count < 1) return;
    auto [begin, end] = m.vocoder.input_frames(first, count, total);
    int frames = end - begin, samples = m.vocoder.samples(count);
    Vocoder::Workspace work(m.vocoder, 1, frames, &resources.workspace());
    auto output = audio.view(m.vocoder.samples(frames), 1, type);
    m.vocoder.forward(work, mel.slice(begin, frames), output, ForwardBatch{1, frames});
    tts::pcm(gpu.stream, output, pcm.ptr<int16_t>());
    gpu.download(host_pcm.data(), pcm.ptr<int16_t>() + m.vocoder.samples(first - begin),
                 size_t(samples) * sizeof(int16_t));
    s.pcm_cursor += count;
    emit(s, callback, host_pcm.data(), samples);
}

int Runner::Impl::flow_window(const std::vector<State *> &rows, const WindowShape &shape,
                              const std::vector<size_t> &members, int first) {
    auto [prefix, accepted, lookahead, context_frames, steps, guidance, mode] = shape;
    const bool offline = mode == SynthesisMode::Offline;
    auto sampling = m.flow.sampling(steps, guidance);
    const int time = prefix + accepted + lookahead;
    bool graph_enabled = std::all_of(members.begin(), members.end(), [&](size_t i) {
        return rows[i]->options.use_cuda_graph;
    });
    int batch = !first ? shapes.select(int(members.size())) : int(members.size());
    if (!batch || batch > window_batch(capacity, time, true))
        batch = int(members.size());
    auto meta_lease = ArenaLease(metadata);
    Metadata upload{metadata, gpu};
    if (batch > capacity || time > max_flow_time)
        throw std::logic_error("CFM descriptors exceed initialized capacity");
    auto &feature_rows = host.feature_rows;
    auto &prompt_rows = host.prompt_rows;
    auto &history_rows = host.history_rows;
    auto &feature_output_rows = host.feature_output_rows;
    auto &value_rows = host.value_rows;
    auto &mel_rows = host.mel_rows;
    auto &prefixes = host.prefixes;
    auto &seeds = host.seeds;
    auto &windows = host.windows;
    auto &condition = host.condition;
    for (auto *v : {&feature_rows, &prompt_rows}) v->clear();
    for (auto *v : {&history_rows, &feature_output_rows, &value_rows}) v->clear();
    mel_rows.clear();
    prefixes.assign(batch, prefix);
    seeds.clear();
    windows.clear();
    condition.mel.clear();
    int mel_offset = 0;
    for (auto index : members) {
        auto &s = *rows[index];
        const int tail = s.cursor ? std::min({m.rolling_frames, prefix, s.history_count}) : 0;
        auto refmel = s.data.reference_mel.view(prefix, m.mel_channels, type);
        auto reffeat = s.data.reference_features.view(prefix, m.feature_channels, type);
        auto features =
            s.data.features.view(m.acoustic.feature_frames(s.count, s.options.speed_factor), m.feature_channels, type);
        int history_frames = m.acoustic.feature_capacity(context);
        auto old = s.data.history.view(history_frames, m.mel_channels, type);
        auto saved_features = s.data.feature_history.view(history_frames, m.feature_channels, type);
        auto output_mel = s.data.offline_mel.view(history_frames, m.mel_channels, type);
        condition.mel.push_back(refmel.slice(0, prefix - tail));
        if (tail)
            condition.mel.push_back(old.slice(s.history_count - tail, tail));

        for (int t = 0; t < time; ++t) {
            if (t < prefix) {
                bool original = t < prefix - tail;
                feature_rows.push_back(
                    original ? reffeat.slice(t, 1).data
                             : (mode == SynthesisMode::Hybrid ? saved_features : features)
                                   .slice(s.cursor - tail + t - (prefix - tail), 1).data);
                prompt_rows.push_back(
                    original ? refmel.slice(t, 1).data
                             : old.slice(s.history_count - tail + t - (prefix - tail), 1).data);
                mel_rows.push_back(mel_offset + t);
            } else {
                feature_rows.push_back(features.slice(s.cursor + t - prefix, 1).data);
                prompt_rows.push_back(nullptr);
                mel_rows.push_back(-1);
            }
            int target = t - prefix;
            bool retain = target >= 0 && target < accepted;
            history_rows.push_back(retain ? old.slice(s.cursor + target, 1).data : nullptr);
            if (mode == SynthesisMode::Hybrid)
                feature_output_rows.push_back(retain ? saved_features.slice(s.cursor + target, 1).data
                                                     : nullptr);
            if (offline)
                value_rows.push_back(retain ? output_mel.slice(s.cursor + target, 1).data : nullptr);
        }
        mel_offset += prefix;
        seeds.push_back(s.options.seed);
        windows.push_back(uint64_t(s.windows));
    }
    feature_rows.resize(size_t(batch) * time, nullptr);
    prompt_rows.resize(feature_rows.size(), nullptr);
    history_rows.resize(feature_rows.size(), nullptr);
    if (mode == SynthesisMode::Hybrid) feature_output_rows.resize(feature_rows.size(), nullptr);
    if (offline) value_rows.resize(feature_rows.size(), nullptr);
    mel_rows.resize(feature_rows.size(), -1);
    seeds.resize(batch, 0);
    windows.resize(batch, 0);
    condition.feature_rows = upload.put(feature_rows);
    if (mode == SynthesisMode::Hybrid)
        gpu.upload(flow_feature_rows.data(), feature_output_rows.data(),
                   feature_output_rows.size() * sizeof(void *));
    condition.retained_feature_rows = mode == SynthesisMode::Hybrid
                                         ? flow_feature_rows.ptr<void *>() : nullptr;
    condition.mel_rows = upload.put(mel_rows);
    gpu.upload(window_prefixes.data(), prefixes.data(), prefixes.size() * sizeof(int));
    auto *dprefix = window_prefixes.ptr<int>();
    auto *dseeds = upload.put(seeds);
    auto *dwindows = upload.put(windows);
    gpu.upload(flow_prompt_rows.data(), prompt_rows.data(), prompt_rows.size() * sizeof(void *));
    gpu.upload(flow_history_rows.data(), history_rows.data(), history_rows.size() * sizeof(void *));
    if (offline)
        gpu.upload(flow_value_rows.data(), value_rows.data(), value_rows.size() * sizeof(void *));
    auto &graphs = resources.graphs();
    auto defaults = m.flow.sampling();
    bool use_flow_graph = !first && !offline && graph_enabled &&
        sampling.steps == defaults.steps && sampling.guidance == defaults.guidance &&
        graphs.can_replay("flow", {batch, time, prefix, context_frames});
    {
        auto lease = resources.lease_workspace();
        Flow::Workspace work(m.flow, use_flow_graph ? capacity : batch,
                            use_flow_graph ? flow_time : time, &resources.workspace());
        {
            Range range("CFM condition");
            m.prepare_flow(work, condition, ForwardBatch{batch, time}, dprefix, dseeds, dwindows, sampling);
        }
        {
            Range range("CFM steps");
            graphs.run("flow", {batch, time, prefix, context_frames}, use_flow_graph, [&] {
                flow_body(work, batch, time, prefix, context_frames, sampling, offline, first);
            });
        }
    }
    if (offline) {
        for (auto index : members) {
            auto &s = *rows[index];
            s.cursor += accepted;
            ++s.windows;
            s.history_count = s.cursor;
        }
    }
    return batch;
}

void Runner::Impl::synthesize_window(const std::vector<State *> &rows,
                                     const std::vector<AudioCallback> &callbacks,
                                     const WindowShape &shape, const std::vector<size_t> &members) {
    auto [prefix, accepted, lookahead, context_frames, steps, guidance, mode] = shape;
    const int time = prefix + accepted + lookahead,
              vocoder_frames = context_frames + accepted + lookahead;
    const int flow_limit = window_batch(capacity, time, true);
    int generated = 0;
    for (size_t first = 0; first < members.size(); first += flow_limit) {
        auto &chunk = host.chunk;
        chunk.assign(members.begin() + first,
                     members.begin() + std::min(members.size(), first + size_t(flow_limit)));
        generated = int(first) + flow_window(rows, shape, chunk, int(first));
    }
    if (mode == SynthesisMode::Offline) return;
    const bool graph_enabled = std::all_of(members.begin(), members.end(), [&](size_t i) {
        return rows[i]->options.use_cuda_graph;
    });
    const int audio_limit = window_batch(capacity, vocoder_frames, false);
    for (size_t audio_first = 0; audio_first < members.size(); audio_first += audio_limit) {
        const int audio_count = int(std::min(members.size() - audio_first, size_t(audio_limit)));
        int audio_batch = !audio_first ? shapes.select(audio_count) : audio_count;
        if (!audio_batch || audio_batch > audio_limit ||
            audio_batch > generated - int(audio_first)) audio_batch = audio_count;
        auto &graphs = resources.graphs();
        bool use_audio_graph = !audio_first && graph_enabled &&
                              graphs.can_replay("audio", {vocoder_frames, audio_batch});
        {
            Range range("vocoder");
            auto lease = resources.lease_workspace();
            graphs.run("audio", {vocoder_frames, audio_batch}, use_audio_graph, [&] {
                Vocoder::Workspace work(m.vocoder, audio_batch, vocoder_frames, &resources.workspace());
                vocode(work, audio_batch, vocoder_frames, int(audio_first));
            });
        }
        gpu.download(host_pcm.data(), pcm.data(),
                     size_t(audio_count) * m.vocoder.samples(vocoder_frames) * sizeof(int16_t));
        const int stride = m.vocoder.samples(vocoder_frames);
        for (int b = 0; b < audio_count; ++b) {
            auto index = members[audio_first + b];
            emit_window(*rows[index], callbacks[index], host_pcm.data() + b * stride,
                        accepted, context_frames);
        }
    }
}

void Runner::Impl::emit_window(State &s, const AudioCallback &callback, int16_t *audio_start,
                              int accepted, int context_frames) {
    const int offset = m.vocoder.samples(context_frames);
    auto *current = audio_start + offset;
    const int samples = m.vocoder.samples(accepted),
              overlap = m.vocoder.samples(s.input.windows.fade);
    const size_t pending = s.pending.size();
    if (pending > size_t(offset))
        throw std::logic_error("PCM fade exceeds downloaded context");
    auto *output = current - pending;
    for (size_t j = 0; j < pending; ++j) {
        float alpha = pending > 1 ? float(j) / (pending - 1) : 1;
        output[j] = int16_t(std::lround(s.pending[j] * (1 - alpha) + output[j] * alpha));
    }
    s.cursor += accepted;
    ++s.windows;
    s.history_count = s.cursor;
    const int total = total_frames(s);
    bool final = s.semantic_done && s.cursor == total;
    int send_samples = final ? samples : samples - overlap;
    const size_t output_samples = pending + send_samples;
    s.pending.assign(current + send_samples, current + samples);
    emit(s, callback, output, output_samples);
    if (s.input.windows.mode == SynthesisMode::Hybrid)
        s.schedule.observe(accepted, output_samples, m.config.at("sample_rate"), now_seconds());
}

void Runner::Impl::synthesize(const std::vector<State *> &rows,
                              const std::vector<AudioCallback> &callbacks) {
    auto &groups = host.groups;
    groups.clear();
    for (size_t i = 0; i < rows.size(); ++i) {
        auto &s = *rows[i];
        int total = total_frames(s);
        int accepted = ready_frames(s);
        // Batch only rows that are ready now. A younger request must never
        // delay audio delivery for an existing request.
        if (!accepted)
            continue;
        int lookahead = std::min(s.input.windows.lookahead(s.cursor), total - s.cursor - accepted);
        groups.push_back({{s.input.reference->frames, accepted, lookahead,
                           s.cursor ? std::min(s.input.windows.left, s.history_count) : 0,
                           s.options.sample_steps, s.options.cfg_rate, s.input.windows.mode}, i});
    }
    // Acoustic self-attention depends on semantic lengths, not CFM window shape.
    // Merge every ready request before partitioning the downstream window groups.
    auto &ready = host.members;
    ready.clear();
    for (const auto &group : groups) ready.push_back(group.second);
    if (!ready.empty()) acoustic_batch(rows, &ready, false);
    // Match the old map's shape order and original row order without node allocation.
    std::sort(groups.begin(), groups.end());
    for (size_t first = 0; first < groups.size();) {
        auto &members = host.members;
        members.clear();
        size_t end = first;
        while (end < groups.size() && groups[end].first == groups[first].first)
            members.push_back(groups[end++].second);
        synthesize_window(rows, callbacks, groups[first].first, members);
        first = end;
    }
    for (size_t i = 0; i < rows.size(); ++i) {
        auto &s = *rows[i];
        if (s.semantic_done &&
            s.cursor == total_frames(s)) {
            if (s.input.windows.mode == SynthesisMode::Offline && !s.finished) {
                synthesize_offline(s, callbacks[i]);
                if (!s.finished && s.pcm_cursor < total_frames(s)) {
                    s.metrics.total_ms = elapsed(s.start);
                    continue;
                }
            }
            emit(s, callbacks[i], s.pending);
            s.pending.clear();
            if (!s.finished && s.segment < s.input.segments.size() &&
                s.metrics.frames < s.options.max_frames) {
                int pause = std::min(s.input.pause_frames, s.options.max_frames - s.metrics.frames);
                while (pause && !s.finished) {
                    const int count = std::min(pause, max_vocoder_time);
                    const size_t samples = size_t(m.vocoder.samples(count));
                    std::fill_n(host_pcm.data(), samples, int16_t{});
                    emit(s, callbacks[i], host_pcm.data(), samples);
                    pause -= count;
                }
                if (!s.finished && s.metrics.frames < s.options.max_frames) {
                    s.input.text = std::move(s.input.segments[s.segment++]);
                    prepare_segment(s);
                } else s.finished = true;
            } else s.finished = true;
        }
        s.metrics.total_ms = elapsed(s.start);
    }
}
void Runner::step(const std::vector<RequestState *> &requests,
                  const std::vector<AudioCallback> &callbacks) {
    auto &r = *impl_;
    CUDA(cudaSetDevice(r.gpu.device));
    ExecutionScope execution(r.gpu);
    if (requests.empty() || requests.size() != callbacks.size())
        throw std::invalid_argument("invalid execution batch");
    auto &rows = r.host.rows;
    auto &decode = r.host.decode;
    rows.clear();
    decode.clear();
    for (auto *request : requests) {
        auto &s = r.state(*request);
        rows.push_back(&s);
        if (!s.finished && !s.semantic_done && !r.ready_frames(s))
            decode.push_back(&s);
    }
    r.advance(decode);
    r.synthesize(rows, callbacks);
}
std::shared_ptr<const Reference> Runner::prepare_reference(ReferencePlan &base) {
    auto &r = *impl_;
    auto &plan = dynamic_cast<ReferenceInput &>(base);
    CUDA(cudaSetDevice(r.gpu.device));
    ExecutionScope execution(r.gpu);
    auto reference = std::make_shared<ReferenceData>();
    {
        auto lease = r.resources.lease_workspace();
        ReferenceWork work(r.m, r.context, &r.resources.workspace());
        r.gpu.upload(work.wave.data(), plan.wave.data(), plan.wave.size() * sizeof(float));
        r.gpu.upload(work.small.data(), plan.hubert_wave.data(),
                     plan.hubert_wave.size() * sizeof(uint16_t));
        r.gpu.upload(work.phones.data(), plan.text.phones.data(),
                     plan.text.phones.size() * sizeof(int));
        auto encoded = r.m.reference.forward(work.encoder,
                                             work.wave.view(int(plan.wave.size()), 1, Type::F32),
                                             work.small.view(int(plan.hubert_wave.size()), 1, r.type));
        int frames = r.m.acoustic.feature_frames(encoded.codes.rows);
        auto features = work.features.view(frames, r.m.feature_channels, r.type);
        r.m.acoustic.forward(work.acoustic, encoded.codes.ptr<int>(), work.phones.ptr<int>(),
                             encoded.style, features,
                             r.m.acoustic.describe(1, encoded.codes.rows, int(plan.text.phones.size())));
        int limit = std::min(frames, encoded.mel.rows), count = std::min(limit, r.m.reference_frames);
        reference->type = r.type;
        reference->frames = count;
        reference->codes.resize(encoded.codes.rows);
        reference->style.resize(encoded.style.numel());
        reference->waveform_condition.resize(r.m.acoustic.waveform_condition_channels());
        reference->mel.resize(size_t(count) * r.m.mel_channels);
        reference->features.resize(size_t(count) * r.m.feature_channels);
        r.gpu.download(reference->codes.data(), encoded.codes.data, encoded.codes.size());
        r.gpu.download(reference->style.data(), encoded.style.data, encoded.style.size());
        r.gpu.download(reference->waveform_condition.data(), work.acoustic.waveform_features.condition.data(),
                       reference->waveform_condition.size() * sizeof(uint16_t));
        r.gpu.download(reference->mel.data(), encoded.mel.slice(limit - count, count).data,
                       reference->mel.size() * sizeof(uint16_t));
        r.gpu.download(reference->features.data(), features.slice(limit - count, count).data,
                       reference->features.size() * sizeof(uint16_t));
    }
    if (!plan.text.bert_ids.empty()) {
        auto lease = r.resources.lease_workspace();
        auto meta_lease = ArenaLease(r.metadata);
        Metadata upload{r.metadata, r.gpu};
        auto &descriptors = r.host.prefill;
        descriptors.clear();
        const int time = descriptors.append_bert(plan.text);
        const auto *ids = upload.put(descriptors.ids);
        const auto bert_rows = descriptors.bert.view();
        const auto *positions = upload.put(bert_rows.row_position, bert_rows.rows);
        ForwardBatch attention(SequenceLayout{nullptr, nullptr,
            upload.put(bert_rows.cu_seqlens, size_t(bert_rows.sequences) + 1), nullptr, int(descriptors.ids.size()),
            descriptors.bert.size(), time});
        attention.positions = positions;
        Bert::Workspace bert(r.m.bert, std::max(1, r.context / 3), r.bert_time,
                              &r.resources.workspace(), r.context);
        Semantic::Workspace semantic(r.m.semantic, 1, r.context, &r.resources.workspace());
        auto features = r.m.bert.forward(bert, ids, attention);
        auto projected = r.m.semantic.encode_features(semantic, features);
        reference->bert_projection.resize(projected.numel());
        r.gpu.download(reference->bert_projection.data(), projected.data, projected.size());
    }
    reference->text = std::move(plan.text);
    return std::make_shared<const Reference>(std::move(plan.transcript), std::move(reference));
}
} // namespace tts::gpt_sovits
