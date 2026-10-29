#include "sampling.h"
#include "audio_kernels.h"
#include "model_runner.h"
#include "generation_state.h"
#include "execution_resources.h"
#include "prepared_batch.h"
#include "prefill.h"
#include "tensor_debug.h"
#include <chrono>
#include <cmath>
#include <fstream>
#include <filesystem>
namespace tts {
using Clock = std::chrono::steady_clock;
static double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
void ModelRunner::depth_frame(GenerationState &s, int batch, int branches,
                              const GenerationOptions &options, bool debug) {
    int lanes = batch * branches;
    auto &g = gpu;
    auto h = s.last.view(lanes, model.backbone.config.hidden),
         l = s.logits.view(lanes, model.lm_head.rows);
    {
        Range r("tts.lm_head");
        g.linear(h, model.lm_head, l);
        tts::sample(
            g.stream, l,
            {model.config.codebooks, model.config.codec.codebook_size, model.config.backbone_eos},
            s.codes.ptr<int>(), s.rng.ptr<uint64_t>(), batch, branches, 0, options.cfg_scale,
            options.do_sample ? options.temperature : 0.f, options.top_k, s.done.ptr<int>(),
            options.top_p, options.cfg_scale_ref, options.cfg_scale_ins, options.repetition_penalty,
            s.seen.ptr<int>());
    }
    if (debug)
        dump_tensor(g, l, ".backbone_logits.bf16");
    Range range("tts.depth");
    auto hidden = s.depth_hidden.view(lanes * 2, model.depth.config.hidden),
         dl = s.depth_logits.view(lanes, model.depth_heads[0].rows);
    auto first = hidden.every_rows(0, 2, lanes);
    auto dh = hidden.every_rows(1, 2, lanes);
    g.linear(h, model.depth_proj, first);
    tts::audio_embedding(g.stream, model.depth_embedding, s.codes.ptr<int>(), dh, batch, branches,
                         model.config.codebooks, 0);
    model.depth.forward(g, s.ds, hidden, nullptr, lanes, 2, 0, 0);
    for (int cb = 1; cb < model.config.codebooks; cb++) {
        if (cb > 1) {
            hidden = s.depth_hidden.view(lanes, model.depth.config.hidden);
            dh = hidden;
            tts::audio_embedding(g.stream, model.depth_embedding, s.codes.ptr<int>(), hidden, batch,
                                 branches, model.config.codebooks, cb - 1);
            model.depth.forward(g, s.ds, hidden, nullptr, lanes, 1, 0, cb);
        }
        g.linear(dh, model.depth_heads[cb - 1], dl);
        if (debug)
            dump_tensor(g, dl, ".depth_" + std::to_string(cb) + ".bf16");
        tts::sample(
            g.stream, dl,
            {model.config.codebooks, model.config.codec.codebook_size, model.config.backbone_eos},
            s.codes.ptr<int>(), s.rng.ptr<uint64_t>(), batch, branches, cb, options.cfg_scale,
            options.depth_do_sample ? options.depth_temperature : 0.f, options.depth_top_k,
            s.done.ptr<int>(), options.depth_top_p, options.cfg_scale_ref, options.cfg_scale_ins);
    }
}
void ModelRunner::next_frame(GenerationState &s, int batch, int branches,
                             const GenerationOptions &options) {
    int lanes = batch * branches;
    {
        Range r("tts.backbone");
        auto input = s.last.view(lanes, model.backbone.config.hidden);
        tts::audio_embedding(gpu.stream, model.audio_embedding, s.codes.ptr<int>(), input, batch,
                             branches, model.config.codebooks);
        model.backbone.forward(gpu, s.bs, input, s.positions.ptr<int>(), lanes, 1);
        gpu.increment(s.positions.ptr<int>(), lanes);
    }
    depth_frame(s, batch, branches, options);
}
int ModelRunner::generate(const std::vector<Request> &requests, const GenerationOptions &opt,
                          AudioCallback callback, Stats &stats) {
    CUDA(cudaSetDevice(gpu.device));
    Range request_range("tts.request");
    stats = {};
    auto start = Clock::now();
    auto prepared = prepare_batch(model.tokenizer, model.config, max_batch, context, requests, opt);
    int batch = prepared.batch, branches = prepared.branches, lanes = prepared.lanes;
    int rows = prepared.storage_rows, text_rows = prepared.text_rows,
        prompt_rows = prepared.prompt_rows;
    gpu.begin_admission();
    model.text.prepare(gpu, text_rows);
    model.backbone.prepare(gpu, prompt_rows);
    model.backbone.prepare(gpu, lanes);
    model.depth.prepare(gpu, lanes);
    model.depth.prepare(gpu, lanes * 2);
    WorkspacePlan plan;
    GenerationState planned(model, context, batch, branches, rows, &plan, resources.kv_view());
    plan.allocate(prefill_workspace_bytes(model, prepared));
    resources.prepare(plan.size, prepared.capacities);
    auto cache_lease = resources.lease_kv(prepared.capacities);
    auto lease = resources.lease_workspace();
    GenerationState s(model, context, batch, branches, rows, &resources.workspace(),
                      resources.kv_view());
    // Only capture-time scalar arguments belong in the key; seeds and positions live in buffers.
    auto &graphs =
        resources.graphs_for({batch, branches, rows, opt.temperature, opt.cfg_scale, opt.top_k,
                              opt.top_p, opt.repetition_penalty, opt.do_sample, opt.depth_top_k,
                              opt.depth_temperature, opt.depth_top_p, opt.depth_do_sample,
                              opt.cfg_scale_ref, opt.cfg_scale_ins, gpu.execution_revision()});
    gpu.zero(s.codes);
    gpu.zero(s.done);
    gpu.zero(s.seen);
    s.cs.reset(gpu);
    std::vector<uint64_t> seeds(batch);
    for (int i = 0; i < batch; i++)
        seeds[i] = opt.seed + uint64_t(i) * 0xd1342543de82ef95ULL;
    gpu.upload(s.rng.data(), seeds.data(), s.rng.size());
    gpu.sync();
    prefill(model, gpu, resources.workspace(), prepared, s);
    stats.prefill_ms = milliseconds(start);
    std::vector<int> ended(batch);
    std::vector<int16_t> pcm(size_t(batch) * model.config.samples_per_frame);
    std::ofstream token_out;
    if (const char *path = std::getenv("TTS_TOKENS_OUT")) {
        token_out.open(std::filesystem::u8path(path), std::ios::binary);
        if (!token_out)
            throw std::runtime_error("cannot create token debug output");
    }
    bool cancelled = false;
    int delivered = 0;
    for (int frame = 0; frame < opt.max_frames; frame++) {
        Range frame_range(frame < 2 ? "tts.frame.setup" : "tts.frame.steady");
        if (frame == 0) {
            if (opt.use_cuda_graph && !std::getenv("TTS_DUMP_PREFIX")) {
                if (!graphs.first_depth.executable)
                    graphs.first_depth.capture(gpu.stream,
                                               [&] { depth_frame(s, batch, branches, opt); });
                graphs.first_depth.run(gpu.stream);
            } else
                depth_frame(s, batch, branches, opt, std::getenv("TTS_DUMP_PREFIX"));
        } else if (opt.use_cuda_graph) {
            if (!graphs.decoder.executable)
                graphs.decoder.capture(gpu.stream, [&] { next_frame(s, batch, branches, opt); });
            graphs.decoder.run(gpu.stream);
        } else
            next_frame(s, batch, branches, opt);
        if (frame == 1)
            dump_tensor(gpu, s.last.view(lanes, model.backbone.config.hidden).slice(0),
                        ".backbone_step1.bf16");
        gpu.download(ended.data(), s.done.data(), batch * 4);
        if (std::all_of(ended.begin(), ended.end(), [](int v) { return v != 0; }))
            break;
        if (token_out.is_open()) {
            std::vector<int> c(batch * model.config.codebooks);
            gpu.download(c.data(), s.codes.data(), c.size() * 4);
            token_out.write(reinterpret_cast<const char *>(c.data()), c.size() * 4);
            if (!token_out)
                throw std::runtime_error("cannot write token debug output");
        }
        {
            Range r("tts.codec");
            if (opt.use_cuda_graph) {
                if (!graphs.codec.executable)
                    graphs.codec.capture(
                        gpu.stream, [&] { model.codec.forward(gpu, s.cs, s.codes.ptr<int>()); });
                graphs.codec.run(gpu.stream);
            } else
                model.codec.forward(gpu, s.cs, s.codes.ptr<int>());
        }
        gpu.download(pcm.data(), s.cs.pcm.data(), pcm.size() * 2);
        if (!delivered)
            stats.first_audio_ms = milliseconds(start);
        {
            Range r("tts.deliver");
            for (int b = 0; b < batch; b++)
                if (!ended[b]) {
                    stats.audio_seconds +=
                        double(model.config.samples_per_frame) / model.config.sample_rate;
                    delivered++;
                    if (!callback(b, pcm.data() + size_t(b) * model.config.samples_per_frame,
                                  model.config.samples_per_frame)) {
                        cancelled = true;
                        break;
                    }
                }
        }
        stats.frames = frame + 1;
        if (cancelled)
            break;
    }
    stats.truncated = !cancelled && stats.frames == opt.max_frames;
    stats.total_ms = milliseconds(start);
    CUDA(cudaGetLastError());
    return cancelled ? 1 : 0;
}
std::vector<int> ModelRunner::encode_audio(const std::vector<float> &mono) {
    CUDA(cudaSetDevice(gpu.device));
    if (mono.size() > size_t(model.config.max_reference_frames) * model.config.samples_per_frame)
        throw std::invalid_argument("reference audio exceeds context");
    resources.prepare(model.encoder.workspace_bytes(mono.size()));
    auto lease = resources.lease_workspace();
    return model.encoder.encode(gpu, resources.workspace(), mono);
}
std::vector<float> ModelRunner::decode_codes(const std::vector<int> &codes) {
    CUDA(cudaSetDevice(gpu.device));
    const auto &c = model.config;
    size_t frames = codes.size() / c.codebooks;
    if (codes.empty() || codes.size() % c.codebooks ||
        frames > size_t(c.codec.transformer.max_positions))
        throw std::runtime_error("codes must contain complete frames within codec position limits");
    for (int code : codes)
        if (code < 0 || code >= c.codec.codebook_size)
            throw std::runtime_error("codec ID out of range");
    WorkspacePlan plan;
    CodecState planned(model.codec, 1, &plan);
    size_t code_bytes = size_t(c.codebooks) * sizeof(int);
    plan.allocate(code_bytes);
    resources.prepare(plan.size);
    auto lease = resources.lease_workspace();
    CodecState state(model.codec, 1, &resources.workspace());
    state.reset(gpu);
    Buffer frame(code_bytes, &resources.workspace());
    std::vector<float> pcm(frames * c.samples_per_frame);
    for (size_t t = 0; t < frames; t++) {
        gpu.upload(frame.data(), codes.data() + t * c.codebooks, code_bytes);
        TensorView out = model.codec.forward(gpu, state, frame.ptr<int>());
        gpu.download(pcm.data() + t * c.samples_per_frame, out.data, out.size());
    }
    return pcm;
}
} // namespace tts
