#pragma once
#include "core/buffer.h"
#include "ops/basic.h"
#include "models/gpt_sovits/dit.h"
#include "core/tensor_batch.h"
#include "core/forward_context.h"
#include <cmath>
#include <cstring>

namespace tts {
class Flow {
    DiT estimator_;
    int mel_channels_;
    float temperature_, guidance_;
    float minimum_, maximum_;
    int rolling_, max_time_;
    static int execution_batch(int batch, float guidance) {
        int branches = guidance > 1.e-5f ? 2 : 1;
        if (batch < 1 || batch > INT32_MAX / branches)
            throw std::invalid_argument("invalid flow batch capacity");
        return batch * branches;
    }

  public:
    FlowSampling sampling(int steps = 0, float guidance = -1) const {
        return FlowSampling::resolve(steps, guidance, {estimator_.steps(), guidance_});
    }
  private:
    struct Workspace {
        DiT::Workspace estimator;
        Buffer noise, velocity, prefixes;
        int max_batch;
        Workspace(const Flow &f, int batch, int time)
            : estimator(f.estimator_, f.execution_batch(batch, 1), time),
              noise(size_t(batch) * time * f.mel_channels_ * bytes(f.estimator_.type())),
              velocity(size_t(estimator.max_batch) * time * f.mel_channels_ *
                           bytes(f.estimator_.type())),
              prefixes(size_t(batch) * sizeof(int)),
              max_batch(batch) {}
        void validate(int batch, int time) const {
            if (batch < 1 || batch > max_batch || time < 1 || time > estimator.max_time)
                throw std::invalid_argument("flow execution exceeds initialized workspace");
        }
    };
  public:
    Flow(Weights &weights, const std::string &prefix, const Json &config, int max_time,
          float minimum, float maximum, int rolling)
        : estimator_(weights, prefix, config, max_time), mel_channels_(config.at("mel_channels")),
          temperature_(config.at("noise_temperature")), guidance_(config.at("cfg_scale")),
          minimum_(minimum), maximum_(maximum), rolling_(rolling), max_time_(max_time) {
        if (!(temperature_ >= 0) || !std::isfinite(temperature_) || !(guidance_ >= 0) ||
            !std::isfinite(guidance_))
            throw std::invalid_argument("invalid flow temperature/guidance");
    }
    TensorBatch forward(const TensorBatch &features, TensorInputs, ForwardContext &) const;
  private:
    void prepare(Workspace &w, const DiT::Condition &condition, const ForwardBatch &input,
                 FlowSampling parameters) const {
        w.validate(input.batch(), input.time());
        estimator_.prepare_condition(w.estimator, condition, input,
                                      parameters.guidance > 1.e-5f);
    }
    void initialize(Workspace &w, const ForwardBatch &input, const int *prompt,
                    const uint64_t *seeds, const uint64_t *windows) const {

        int batch = input.batch(), time = input.time();
        w.validate(batch, time);
        ops::gaussian_noise(
                            w.noise.view(batch * time, mel_channels_, estimator_.type()), time,
                            prompt, seeds, windows, temperature_);
    }
    void advance(Workspace &workspace, const ForwardBatch &input, FlowSampling parameters,
                 const ops::EulerOutput &output, ForwardContext &context) const {
        int batch = input.batch(), time = input.time();
        workspace.validate(batch, time);
        auto noise = workspace.noise.view(batch * time, mel_channels_, estimator_.type());
        int effective_batch = execution_batch(batch, parameters.guidance);
        auto velocity = workspace.velocity.view(effective_batch * time, mel_channels_, estimator_.type());
        auto positive = velocity.slice(0, noise.rows);
        auto negative = effective_batch > batch ? velocity.slice(noise.rows, noise.rows) : TensorView{};
        auto update = [&](const ops::EulerOutput &target) {
            ops::euler_step(noise, positive, negative, time, workspace.prefixes.ptr<int>(),
                1.f / parameters.steps, parameters.guidance > 1.e-5f ? parameters.guidance : 0.f, target);
        };
        int guidance_bits;
        std::memcpy(&guidance_bits, &parameters.guidance, sizeof(guidance_bits));
        context.run(this, "flow", {batch, time, parameters.steps, guidance_bits},
            {workspace.noise.data(), workspace.prefixes.data()}, [&] {
                for (int index = 0; index < parameters.steps; ++index) {
                    estimator_.forward(workspace.estimator, noise, velocity, input, index, parameters.steps);
                    if (index + 1 < parameters.steps) update({});
                }
            });
        if (context.mode == ForwardContext::Execute) update(output);
    }
};
} // namespace tts
