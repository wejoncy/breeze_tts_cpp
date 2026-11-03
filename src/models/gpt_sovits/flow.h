#pragma once
#include "models/gpt_sovits/dit.h"
#include <cmath>

namespace tts::gpt_sovits {
class Flow {
    DiT estimator_;
    int mel_channels_;
    float temperature_, guidance_;
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
    struct Workspace {
        DiT::Workspace estimator;
        Buffer noise, velocity;
        int max_batch;
        Workspace(const Flow &f, int batch, int time, BufferAllocator *allocator)
            : estimator(f.estimator_, f.execution_batch(batch, 1), time, allocator),
              noise(size_t(batch) * time * f.mel_channels_ * bytes(f.estimator_.type()), allocator),
              velocity(size_t(estimator.max_batch) * time * f.mel_channels_ *
                           bytes(f.estimator_.type()),
                       allocator),
              max_batch(batch) {}
        void validate(int batch, int time) const {
            if (batch < 1 || batch > max_batch || time < 1 || time > estimator.max_time)
                throw std::invalid_argument("flow execution exceeds initialized workspace");
        }
    };
    Flow(Weights &weights, const Json &config, int max_time)
        : estimator_(weights, config, max_time), mel_channels_(config.at("mel_channels")),
          temperature_(config.at("noise_temperature")), guidance_(config.at("cfg_scale")) {
        if (!(temperature_ >= 0) || !std::isfinite(temperature_) || !(guidance_ >= 0) ||
            !std::isfinite(guidance_))
            throw std::invalid_argument("invalid flow temperature/guidance");
    }
    const DiT &estimator() const {
        return estimator_;
    }
    void prepare(Workspace &w, const DiT::Condition &condition, const ForwardBatch &input,
                 FlowSampling parameters) const {
        w.validate(input.batch(), input.time());
        estimator_.prepare_condition(w.estimator, condition, input,
                                      parameters.guidance > 1.e-5f);
    }
    void initialize(Workspace &w, const ForwardBatch &input, const int *prompt,
                    const uint64_t *seeds, const uint64_t *windows) const {
        auto &gpu = backend_for(w.noise.device());
        int batch = input.batch(), time = input.time();
        w.validate(batch, time);
        ops::gaussian_noise(gpu.stream,
                            w.noise.view(batch * time, mel_channels_, estimator_.type()), time,
                            prompt, seeds, windows, temperature_);
    }
    void step(Workspace &w, const ForwardBatch &input, const int *prompt, int index,
              FlowSampling parameters, const ops::EulerOutput &out = {}) const {
        auto &gpu = backend_for(w.noise.device());
        int batch = input.batch(), time = input.time();
        w.validate(batch, time);
        auto x = w.noise.view(batch * time, mel_channels_, estimator_.type());
        int effective_batch = execution_batch(batch, parameters.guidance);
        auto velocity = w.velocity.view(effective_batch * time, mel_channels_, estimator_.type());
        estimator_.forward(w.estimator, x, velocity, input, index, parameters.steps);
        auto v = velocity.slice(0, x.rows);
        auto negative = effective_batch > batch ? velocity.slice(x.rows, x.rows) : TensorView{};
        ops::euler_step(gpu.stream, x, v, negative, time, prompt, 1.f / parameters.steps,
                        parameters.guidance > 1.e-5f ? parameters.guidance : 0.f, out);
    }
};
} // namespace tts::gpt_sovits
