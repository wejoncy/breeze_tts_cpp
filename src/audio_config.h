#pragma once
#include "transformer_config.h"

namespace tts {
struct EncoderConfig {
    TransformerConfig transformer;
    std::vector<int> ratios;
    int filters, residual_layers, residual_kernel, dilation_growth, compress;
    int first_kernel, last_kernel, stride, downsample_stride;
    int codebooks, semantic_codebooks, codebook_size, codebook_dim;
};
struct CodecConfig {
    TransformerConfig transformer;
    std::vector<int> upsampling_ratios, upsample_rates;
    int latent_dim, codebook_dim, decoder_dim, samples_per_frame;
    int codebooks, semantic_codebooks, codebook_size;
};
} // namespace tts
