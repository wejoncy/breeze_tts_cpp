#pragma once
#include "models/gpt_sovits/relative_encoder.h"
#include "models/gpt_sovits/wavenet.h"
#include "models/gpt_sovits/acoustic_rate.h"
#include "io/json.h"
namespace tts::gpt_sovits {
class Acoustic {
    Tensor projected_codebook_;
    TensorView phones_;
    RelativeEncoder ssl_encoder_, text_encoder_, output_encoder_;
    Linear content_, text_, query_, key_value_, attention_out_, content_out_, bridge_;
    WaveNet waveform_features_;
    int heads_;
    AcousticRate rate_;
    Type type_;

  public:
    struct Capacity {
        int sequences, codes, phones, frames;
    };
    struct Batch {
        SequenceLayout phones{}, hidden{}, resized{}, features{};
        const AttentionSequence *cross_attention = nullptr;
        const int *const *phone_ids = nullptr, *const *code_ids = nullptr;
        const void *const *styles = nullptr, *const *conditions = nullptr,
                         *const *text_keys = nullptr, *const *text_values = nullptr;
        void *const *text_outputs = nullptr, *const *feature_outputs = nullptr;
        bool resample = false;
        int valid_frames = 0;
    };
    struct Workspace {
        RelativeEncoder::Workspace ssl_encoder, text_encoder, output_encoder;
        WaveNet::Workspace waveform_features;
        Buffer ssl, text, content, projected_text, query, key_value, attended, update, lse, features;
        int max_batch, max_codes, max_phones, max_frames;
        Workspace(const Acoustic &, int batch, int codes, int phones, BufferAllocator *,
                  int frames = 0);
        Workspace(const Acoustic &, Capacity, BufferAllocator *);
    };
    Acoustic(Weights &, const Json &);
    int feature_frames(int codes, double speed = 1.) const {
        return rate_.frames(codes, speed);
    }
    int codes_for_frames(int frames, double speed, int limit) const {
        return rate_.codes_for_frames(frames, speed, limit);
    }
    int feature_capacity(int codes) const {
        return rate_.capacity(codes);
    }
    double capacity_speed(int codes) const {
        return rate_.capacity_speed(codes);
    }
    int text_condition_channels() const {
        return key_value_.output_channels();
    }
    int waveform_condition_channels() const { return waveform_features_.condition_channels(); }
    int hidden_frames(int codes, double speed = 1.) const { return rate_.hidden_frames(codes, speed); }
    int valid_frames(int codes, double speed = 1.) const { return rate_.valid_frames(codes, speed); }
    void prepare(Workspace &) const;
    Batch describe(int batch, int codes, int phones, double speed = 1.) const;
    // Text is invariant across streaming windows; callers may retain this K/V
    // projection in request storage instead of re-encoding it for every window.
    void prepare_text(Workspace &, const int *phones, TensorView condition, const Batch &) const;
    void forward(Workspace &, const int *codes, const int *phones, TensorView style,
                 TensorView features, const Batch &, TensorView text_condition = {}) const;
};
} // namespace tts::gpt_sovits
