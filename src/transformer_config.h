#pragma once
#include "tensor.h"
#include "quantization.h"
#include <vector>

namespace tts {
struct TransformerConfig {
    int hidden, layers, heads, kvheads, dim, intermediate;
    float eps, theta;
    bool gemma = false, causal = true, qknorm = false;
    int window = 0, rope_type = 0;
    float factor = 1;
    Type type = Type::BF16;
    int max_positions = 0;
    float full_theta = 0, full_factor = 1;
    float rope_low_wave = 0, rope_high_wave = 0;
    std::vector<bool> full_attention;
    Quantization quantization = Quantization::None;
};
} // namespace tts
