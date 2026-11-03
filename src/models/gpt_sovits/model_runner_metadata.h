#pragma once
#include "models/gpt_sovits/model.h"
#include "models/gpt_sovits/request_planner.h"
#include "core/sequence_builder.h"
#include "runtime/device.h"

namespace tts::gpt_sovits::runner_detail {
// Small explicit host-to-device metadata uploads. No tensor cast, packing or
// device-to-device copy kernel is introduced to move a request between batches.
struct Metadata {
    DeviceArena &arena;
    Backend &gpu;
    template <class T> T *put(const T *values, size_t count) {
        auto *device = static_cast<T *>(arena.allocate(count * sizeof(T)));
        if (count) gpu.upload(device, values, count * sizeof(T));
        return device;
    }
    template <class T> T *put(const std::vector<T> &values) {
        return put(values.data(), values.size());
    }
    SequenceLayout put(const SequenceBuilder &builder) {
        const auto h = builder.view();
        if (builder.uniform()) return SequenceLayout::uniform(h.sequences, h.max_length);
        return {put(h.row_batch, h.rows), put(h.row_position, h.rows),
                put(h.cu_seqlens, size_t(h.sequences) + 1), put(h.lengths, h.sequences),
                h.rows, h.sequences, h.max_length};
    }
    static size_t capacity(size_t batch, size_t context, size_t bert_time, size_t blocks,
                           size_t flow_time) {
        WorkspacePlan prefill, window;
        size_t rows = batch * context, bert_rows = rows;
        for (size_t count : {bert_rows, bert_rows, // BERT IDs and positions
                             rows, rows,           // phones and reference codes
                             rows, rows, rows,     // phone destination, position, BERT row
                             rows, rows,           // code destination and position
                             rows, rows,           // packed sequence and position
                             batch + 1, batch, batch, batch * blocks})
            prefill.allocate(count * sizeof(int));
        for (size_t count : {std::max<size_t>(1, bert_rows / 3), batch, batch})
            prefill.allocate(count * sizeof(AttentionSequence));
        prefill.allocate(rows * sizeof(void *)); // borrowed reference conditioning rows
        size_t frames = batch * flow_time;
        window.allocate(frames * sizeof(void *)); // feature row pointers
        window.allocate(frames * sizeof(int)); // conditioning mel rows
        for (int i = 0; i < 2; ++i)            // seeds and per-request RNG subsequences
            window.allocate(batch * sizeof(uint64_t));
        return std::max(prefill.size, window.size);
    }
};
struct AcousticMetadata {
    SequenceBuilder phones, hidden, resized, features;
    std::vector<AttentionSequence> cross_attention;
    std::vector<const int *> phone_ids, code_ids;
    std::vector<const void *> styles, conditions, keys, values;
    std::vector<void *> text_outputs, feature_outputs;
    std::vector<size_t> members;
    void reserve(const Acoustic &a, Acoustic::Capacity cap) {
        phones.reserve(cap.sequences, cap.phones); hidden.reserve(cap.sequences, a.hidden_frames(cap.codes));
        resized.reserve(cap.sequences, a.hidden_frames(cap.codes) > cap.frames ? a.hidden_frames(cap.codes) : cap.frames);
        features.reserve(cap.sequences, cap.frames);
        cross_attention.reserve(cap.sequences);
        phone_ids.reserve(cap.phones); code_ids.reserve(cap.codes);
        for (auto *v : {&styles, &conditions, &keys, &values}) v->reserve(cap.sequences);
        text_outputs.reserve(cap.phones); feature_outputs.reserve(cap.frames); members.reserve(cap.sequences);
    }
    void clear() {
        for (auto *v : {&phones, &hidden, &resized, &features}) v->clear();
        cross_attention.clear();
        phone_ids.clear(); code_ids.clear(); styles.clear(); conditions.clear(); keys.clear(); values.clear();
        text_outputs.clear(); feature_outputs.clear(); members.clear();
    }
    Acoustic::Batch bind(Metadata &upload, bool text_only) const {
        auto complete = [](const SequenceBuilder &builder) {
            const auto rows = builder.view();
            for (int i = 0; i < rows.sequences; ++i)
                if (rows.lengths[i] != rows.cu_seqlens[i + 1] - rows.cu_seqlens[i])
                    throw std::invalid_argument("relative attention requires complete sequence extents");
        };
        complete(phones);
        if (!text_only) {
            complete(hidden);
            size_t batch = size_t(phones.size());
            if (size_t(hidden.size()) != batch || size_t(resized.size()) != batch ||
                size_t(features.size()) != batch || styles.size() != batch || conditions.size() != batch ||
                keys.size() != batch || values.size() != batch || cross_attention.size() != batch)
                throw std::invalid_argument("packed acoustic domain membership mismatch");
            for (size_t i = 0; i < batch; ++i)
                if (!styles[i] || !conditions[i] || !keys[i] || !values[i] ||
                    cross_attention[i].query_offset != hidden.view().cu_seqlens[i] ||
                    cross_attention[i].query_length != hidden.view().lengths[i] ||
                    cross_attention[i].key_length != phones.view().lengths[i])
                    throw std::invalid_argument("invalid acoustic cross-attention sequence");
        }
        Acoustic::Batch out;
        out.phones = upload.put(phones); out.phone_ids = upload.put(phone_ids);
        out.text_outputs = upload.put(text_outputs);
        if (text_only) return out;
        out.hidden = upload.put(hidden); out.resized = upload.put(resized); out.features = upload.put(features);
        out.code_ids = upload.put(code_ids); out.styles = upload.put(styles); out.conditions = upload.put(conditions);
        out.text_keys = upload.put(keys); out.text_values = upload.put(values);
        out.cross_attention = upload.put(cross_attention);
        out.feature_outputs = upload.put(feature_outputs); out.resample = !hidden.same_extents(resized);
        return out;
    }
    static size_t transfer_bytes(const Acoustic &a, int batch, int context, bool text_only) {
        // Splitting each request separately maximizes upload-alignment overhead.
        // This is a host snapshot bound, not a device workspace allocation.
        size_t one = bytes(a, {1, text_only ? 0 : context, context,
                               text_only ? 0 : a.feature_capacity(context)});
        if (batch < 1 || one > SIZE_MAX / size_t(batch))
            throw std::overflow_error("acoustic transfer capacity");
        return one * size_t(batch);
    }
    static size_t bytes(const Acoustic &a, Acoustic::Capacity cap) {
        WorkspacePlan plan;
        for (int rows : {cap.phones, a.hidden_frames(cap.codes), cap.frames, cap.frames}) {
            plan.allocate(size_t(rows) * sizeof(int)); plan.allocate(size_t(rows) * sizeof(int));
            plan.allocate((size_t(cap.sequences) + 1) * sizeof(int)); plan.allocate(size_t(cap.sequences) * sizeof(int));
        }
        for (int rows : {cap.codes, cap.phones, cap.phones, cap.frames}) plan.allocate(size_t(rows) * sizeof(void *));
        for (int i = 0; i < 4; ++i) plan.allocate(size_t(cap.sequences) * sizeof(void *));
        plan.allocate(size_t(cap.sequences) * sizeof(AttentionSequence));
        return plan.size;
    }
};
// Fixed-capacity descriptors let packed prefill graphs reuse addresses while
// every sequence keeps its own lengths, positions and page table.
struct PrefillMetadata {
    std::vector<int> ids, phones, codes, phone_rows, phone_positions, phone_bert,
        code_output, code_positions, last, tables;
    SequenceBuilder bert, semantic;
    std::vector<AttentionSequence> prefixes, suffixes;
    std::vector<const void *> reference_rows;
    ForwardBatch bert_context;
    Semantic::PrefillBatch input{};
    const int *device_ids = nullptr;
    void reserve(size_t batch, size_t context, size_t blocks) {
        for (auto *v : {&ids, &phones, &codes, &phone_rows, &phone_positions,
                        &phone_bert, &code_output, &code_positions})
            v->reserve(batch * context);
        last.reserve(batch);
        tables.reserve(batch * blocks);
        bert.reserve(int(std::max<size_t>(1, batch * context / 3)), int(batch * context));
        semantic.reserve(int(batch), int(batch * context));
        prefixes.reserve(batch);
        suffixes.reserve(batch);
        reference_rows.reserve(batch * context);
    }
    void clear() {
        for (auto *v : {&ids, &phones, &codes, &phone_rows, &phone_positions,
                        &phone_bert, &code_output, &code_positions,
                        &last, &tables}) v->clear();
        for (auto *v : {&prefixes, &suffixes}) v->clear();
        bert.clear();
        semantic.clear();
        reference_rows.clear();
        bert_context = {};
        input = {};
        device_ids = nullptr;
    }
    int append_bert(const TextInput &text) {
        int maximum = 0;
        ids.insert(ids.end(), text.bert_ids.begin(), text.bert_ids.end());
        for (size_t i = 1; i < text.bert_offsets.size(); ++i) {
            int begin = text.bert_offsets[i - 1], count = text.bert_offsets[i] - begin;
            bert.append(count, count);
            maximum = std::max(maximum, count);
        }
        return maximum;
    }
    void bind(Metadata &upload, int batch, int context, int bert_time, int blocks,
              int bert_rows, int semantic_rows, KVCacheView cache) {
        const size_t b = size_t(batch), r = b * context, br = r;
        if (ids.size() > br || size_t(bert.size()) > std::max<size_t>(1, br / 3))
            throw std::invalid_argument("BERT batch exceeds initialized token capacity");
        ids.resize(br);
        const auto bert_host = bert.padded_view(int(br), int(std::max<size_t>(1, br / 3)));
        phones.resize(r);
        codes.resize(r);
        phone_rows.resize(r, -1);
        phone_positions.resize(r);
        phone_bert.resize(r, -1);
        code_output.resize(r, -1);
        code_positions.resize(r);
        const auto semantic_host = semantic.padded_view(int(r), batch);
        last.resize(b);
        tables.resize(b * blocks);
        prefixes.resize(b);
        suffixes.resize(b);
        reference_rows.resize(r, nullptr);
        device_ids = upload.put(ids);
        bert_context.positions = upload.put(bert_host.row_position, br);
        bert_context.sequence = {nullptr, nullptr, upload.put(bert_host.cu_seqlens, size_t(bert_host.sequences) + 1), nullptr,
                                 bert_rows, 2 * batch, std::min(bert_time, bert_rows)};
        input.phones = upload.put(phones);
        input.codes = upload.put(codes);
        input.phone_rows.output_rows = upload.put(phone_rows);
        input.phone_rows.position_ids = upload.put(phone_positions);
        input.phone_rows.residual_rows = upload.put(phone_bert);
        input.phone_rows.residual_sources = upload.put(reference_rows);
        input.phone_rows.input_rows = semantic_rows;
        input.audio_rows.output_rows = upload.put(code_output);
        input.audio_rows.position_ids = upload.put(code_positions);
        input.audio_rows.input_rows = semantic_rows;
        input.context.sequence = {upload.put(semantic_host.row_batch, r),
                                  upload.put(semantic_host.row_position, r),
                                  upload.put(semantic_host.cu_seqlens, b + 1), nullptr, semantic_rows,
                                  batch, std::min(context, semantic_rows)};
        input.context.bidirectional = upload.put(prefixes);
        input.context.causal = upload.put(suffixes);
        input.context.output_rows = upload.put(last);
        input.context.cache = cache;
        input.context.cache.layout.block_table = upload.put(tables);
        input.context.capacity = context;
    }
};
} // namespace tts::gpt_sovits::runner_detail
