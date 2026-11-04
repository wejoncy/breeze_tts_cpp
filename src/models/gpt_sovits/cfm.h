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
    void update(TensorView noise, TensorView velocity, TensorView prefixes, int time,
                 FlowSampling parameters, const ops::EulerOutput &output = {}) const {
        const auto positive = velocity.slice(0, noise.rows);
        const auto negative = velocity.rows > noise.rows ? velocity.slice(noise.rows, noise.rows) : TensorView{};
        ops::euler_step(noise, positive, negative, time, prefixes.ptr<int>(),
            1.f / parameters.steps, parameters.guidance > 1.e-5f ? parameters.guidance : 0.f, output);
    }
    TensorView advance(TensorView noise, TensorView condition, TensorView prefixes,
                        ForwardContext &context, FlowSampling parameters) const {
        const int batch = context.batch.batch(), time = context.batch.time();
        auto velocity = context.tensor({execution_batch(batch, parameters.guidance) * time, mel_channels_},
                                       estimator_.type());
        if (context.mode == ForwardContext::Plan || context.mode == ForwardContext::Prepare)
            return estimator_.forward(noise, condition, velocity, context, 0, parameters.steps);
        auto execution = context;
        execution.mode = ForwardContext::Execute;
        int guidance_bits;
        std::memcpy(&guidance_bits, &parameters.guidance, sizeof(guidance_bits));
        context.run(this, "flow", {batch, time, parameters.steps, guidance_bits},
            {noise.data, prefixes.data, condition.data, velocity.data}, [&] {
                for (int index = 0; index < parameters.steps; ++index) {
                    estimator_.forward(noise, condition, velocity, execution, index, parameters.steps);
                    if (index + 1 < parameters.steps) update(noise, velocity, prefixes, time, parameters);
                }
            });
        return velocity;
    }
};
} // namespace tts
