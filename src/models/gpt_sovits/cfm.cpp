#include "models/gpt_sovits/cfm.h"
#include "layers/input_metadata.h"
#include <map>
#include <numeric>
#include <tuple>

namespace tts {
TensorBatch Flow::forward(const TensorBatch &feature_sources, TensorInputs inputs, ForwardContext &context) const {
    const int prefix_capacity = context.state["reference_mel"][0].rows;
    const int maximum_time = std::min(max_time_, prefix_capacity + context.max_time);
    const int64_t maximum_rows = int64_t(context.max_batch) * maximum_time * 2;
    if (maximum_rows < 1 || maximum_rows > INT32_MAX)
        throw std::invalid_argument("flow row capacity exceeds tensor dimensions");
    auto output = context.tensor({context.max_batch * context.max_time, mel_channels_}, estimator_.type());
    TensorBatch result;
    if (context.initialize("flow", {context.max_batch, maximum_time, context.max_batch * maximum_time},
        [&](ForwardContext &region) {
            const bool preparing = region.mode == ForwardContext::Prepare;
            const int batch = preparing ? 1 : region.max_batch, time = preparing ? 1 : region.max_time;
            auto invocation = region.with_sequence(SequenceLayout::uniform(batch, time));
            invocation.max_rows = int(maximum_rows);
            auto noise = invocation.tensor({batch * time, mel_channels_}, estimator_.type());
            auto prefixes = invocation.tensor({batch, 1}, Type::I32);
            auto parameters = sampling();
            if (region.mode == ForwardContext::Plan || region.mode == ForwardContext::Prepare)
                parameters.guidance = std::max(parameters.guidance, 1.f);
            auto condition = estimator_.forward({}, invocation, parameters.guidance > 1.e-5f);
            {
                Tensor::Scope noise_inputs;
                for (int field = 0; field < 2; ++field)
                    Tensor::reserve(size_t(batch) * sizeof(uint64_t));
            }
            advance(noise, condition, prefixes, invocation, parameters);
            for (int field = 0; field < 3; ++field)
                Tensor::reserve(size_t(batch * time) * sizeof(void *));
        })) {
        std::vector<int> members(context.max_batch);
        std::iota(members.begin(), members.end(), 0);
        result.chunks.push_back({output, std::move(members), context.max_time});
        result.rows = context.max_batch;
        return result;
    }
    using Shape = std::tuple<int, int, int, int, int, float, int, int>;
    std::map<Shape, std::vector<int>> groups;
    const int batch = inputs["generated"].rows;
    result.rows = batch;
    for (int row = 0; row < batch; ++row)
        groups[{inputs["reference_frames"].ptr<int>()[row], inputs["frames"].ptr<int>()[row],
                inputs["future_frames"].ptr<int>()[row], inputs["history_frames"].ptr<int>()[row],
                inputs["steps"].ptr<int>()[row], inputs["guidance"].ptr<float>()[row],
                inputs["retain_features"].ptr<int>()[row],
                inputs["defer_audio"].ptr<int>()[row]}].push_back(row);
    const auto reference_mel = context.state["reference_mel"];
    const auto reference_features = context.state["reference_features"];
    std::vector<TensorView> features(batch);
    for (const auto &chunk : feature_sources.chunks) {
        if (chunk.rows.size() != 1) throw std::invalid_argument("flow requires per-sequence acoustic features");
        features.at(chunk.rows.front()) = chunk.values;
    }
    const auto history = context.state["history"];
    const auto saved_features = context.state["feature_history"], offline_mel = context.state["offline_mel"];
    int written = 0;
    for (const auto &group : groups) {
        const auto [prefix, accepted, future, left, steps, guidance, retain_features, defer_audio] = group.first;
        const auto &members = group.second;
        auto deferred = [&] {
            for (int row : members) {
                const int total = inputs["available_frames"].ptr<int>()[row];
                const int cursor = inputs["audio_rendered"].ptr<int>()[row];
                if (inputs["rendered"].ptr<int>()[row] + accepted == total && cursor < total)
                    result.chunks.push_back({offline_mel[row].slice(0, total), {row}, total,
                                             cursor, total - cursor});
            }
        };
        if (!accepted && defer_audio) { deferred(); continue; }
        const int time = prefix + accepted + future, frames = left + accepted + future;
        if (prefix < 1 || accepted < 1 || left < 0 || left > prefix || time > maximum_time ||
            frames > context.max_time || written + int(members.size()) * frames > output.rows)
            throw std::invalid_argument("flow window exceeds its declared tensor capacity");
        auto destination = output.slice(written, int(members.size()) * frames);
        Tensor::Scope scratch;
        auto invocation = context.with_sequence(SequenceLayout::uniform(int(members.size()), time));
        invocation.max_rows = int(maximum_rows);
        auto noise = invocation.tensor({int(members.size()) * time, mel_channels_}, estimator_.type());
        auto prefix_tensor = invocation.tensor({int(members.size()), 1}, Type::I32);
        DiT::Condition condition;
        std::vector<const void *> feature_rows, prompt_rows;
        std::vector<void *> history_rows, retained_features, value_rows;
        std::vector<int> mel_rows, prefixes(members.size(), prefix);
        std::vector<uint64_t> seeds, windows;
        int mel_offset = 0;
        for (int row : members) {
            const int cursor = inputs["rendered"].ptr<int>()[row];
            const int tail = cursor ? std::min({rolling_, prefix, cursor}) : 0;
            const bool hybrid = retain_features, offline = defer_audio;
            condition.mel.push_back(reference_mel[row].slice(0, prefix - tail));
            if (tail) condition.mel.push_back(history[row].slice(cursor - tail, tail));
            for (int position = 0; position < time; ++position) {
                if (position < prefix) {
                    const bool original = position < prefix - tail;
                    feature_rows.push_back(original ? reference_features[row].slice(position).data
                        : (hybrid ? saved_features[row] : features[row]).slice(cursor - tail + position - (prefix - tail)).data);
                    prompt_rows.push_back(original ? reference_mel[row].slice(position).data
                        : history[row].slice(cursor - tail + position - (prefix - tail)).data);
                    mel_rows.push_back(mel_offset + position);
                } else {
                    feature_rows.push_back(features[row].slice(cursor + position - prefix).data);
                    prompt_rows.push_back(nullptr);
                    mel_rows.push_back(-1);
                }
                const int target = position - prefix;
                const bool retain = target >= 0 && target < accepted;
                history_rows.push_back(retain ? history[row].slice(cursor + target).data : nullptr);
                if (hybrid) retained_features.push_back(retain ? saved_features[row].slice(cursor + target).data : nullptr);
                if (offline) value_rows.push_back(retain ? offline_mel[row].slice(cursor + target).data : nullptr);
            }
            mel_offset += prefix;
            seeds.push_back(inputs["seed"].ptr<uint64_t>()[row]);
            windows.push_back(inputs["windows"].ptr<uint64_t>()[row]);
        }
        auto upload = [&](const void *data, int rows, Type type) {
            return upload_metadata(context, data, {rows, 1}, type);
        };
        condition.feature_rows = TensorView(feature_rows.data(), int(feature_rows.size()), 1, Type::I64);
        condition.mel_rows = TensorView(mel_rows.data(), int(mel_rows.size()), 1, Type::I32);
        if (!retained_features.empty())
            condition.retained_feature_rows = TensorView(retained_features.data(), int(retained_features.size()), 1, Type::I64);
        ops::upload_bytes(prefix_tensor, prefixes.data(), prefixes.size() * sizeof(int));
        const auto parameters = sampling(steps, guidance);
        auto conditioned = estimator_.forward(condition, invocation, parameters.guidance > 1.e-5f);
        {
            Tensor::Scope noise_inputs;
            const auto seed = upload(seeds.data(), int(seeds.size()), Type::I64);
            const auto window = upload(windows.data(), int(windows.size()), Type::I64);
            ops::gaussian_noise(noise, time, prefix_tensor.ptr<int>(), seed.ptr<uint64_t>(), window.ptr<uint64_t>(), temperature_);
        }
        auto velocity = advance(noise, conditioned, prefix_tensor, invocation, parameters);
        ops::EulerOutput target{destination, prefix - left, frames, minimum_, maximum_,
            upload(prompt_rows.data(), int(prompt_rows.size()), Type::I64).ptr<const void *>(),
            upload(history_rows.data(), int(history_rows.size()), Type::I64).ptr<void *>(),
            value_rows.empty() ? nullptr : upload(value_rows.data(), int(value_rows.size()), Type::I64).ptr<void *>()};
        update(noise, velocity, prefix_tensor, time, parameters, target);
        if (defer_audio) deferred();
        else result.chunks.push_back({destination, members, frames});
        written += int(members.size()) * frames;
    }
    return result;
}
} // namespace tts
