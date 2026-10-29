"""Development-only alternate audio configurations, checked against upstream PyTorch.

Requires the original model's Python dependencies. Fixtures stay in --output;
no weights, audio or Python packages are needed to build the native test binary.
"""

import argparse
import copy
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import MimiConfig
from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2DecoderConfig,
)
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Encoder,
    Qwen3TTSTokenizerV2Decoder,
)


def prepare(model, output):
    source = json.loads((model / "config.json").read_text())
    audio = json.loads((model / "audio_tokenizer/config.json").read_text())
    torch.set_num_threads(2)
    cases = []
    for index, (ratios, layers, residuals, filters, books, size, samples) in enumerate(
        [
            ([3, 2], 2, 2, 12, 4, 64, 37),
            ([2, 3, 2], 1, 3, 10, 6, 128, 113),
            ([3, 2], 3, 1, 16, 3, 32, 1),
        ]
    ):
        torch.manual_seed(319 + index)
        directory = output / f"case-{index}"
        (directory / "audio_tokenizer").mkdir(parents=True, exist_ok=True)
        c, a = copy.deepcopy(source), copy.deepcopy(audio)
        frame_size = int(np.prod(ratios)) * 2
        c.update(
            num_codebooks=books,
            audio_num_codebooks=books,
            audio_vocab_size=size + 3,
            vocab_size=size + 3,
            codebook_pad_token_id=size + 2,
        )
        c["depth_decoder_config"]["vocab_size"] = size + 3
        a.update(
            encoder_valid_num_quantizers=books,
            input_sample_rate=12000,
            output_sample_rate=12000,
            encode_downsample_rate=frame_size,
            decode_upsample_rate=frame_size,
        )
        ec, dc = a["encoder_config"], a["decoder_config"]
        ec.update(
            hidden_size=96,
            num_attention_heads=3,
            num_key_value_heads=3,
            head_dim=32,
            intermediate_size=192,
            num_hidden_layers=layers,
            num_filters=filters,
            num_residual_layers=residuals,
            dilation_growth_rate=3,
            upsampling_ratios=ratios,
            kernel_size=5,
            last_kernel_size=5,
            residual_kernel_size=5,
            compress=2,
            codebook_size=size,
            codebook_dim=24,
            num_quantizers=books,
            num_semantic_quantizers=1,
            sampling_rate=12000,
            max_position_embeddings=256,
            _frame_rate=12000 / frame_size,
            norm_eps=1e-4,
            rope_theta=25000,
            upsample_groups=96,
            vector_quantization_hidden_dimension=24,
        )
        dc.update(
            hidden_size=64,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=32,
            intermediate_size=128,
            num_hidden_layers=layers,
            latent_dim=96,
            decoder_dim=128,
            codebook_dim=48,
            codebook_size=size,
            num_quantizers=books,
            upsample_rates=ratios,
            upsampling_ratios=[2],
            max_position_embeddings=256,
            sliding_window=8,
            rope_theta=20000,
        )
        encoder_config = MimiConfig(**ec)
        encoder_config._attn_implementation = "sdpa"
        decoder_config = Qwen3TTSTokenizerV2DecoderConfig(**dc)
        decoder_config._attn_implementation = "sdpa"
        encoder = Qwen3TTSTokenizerV2Encoder(encoder_config).eval()
        decoder = Qwen3TTSTokenizerV2Decoder(decoder_config).eval()
        with torch.no_grad():
            for module in (encoder, decoder):
                for name, value in module.state_dict().items():
                    if name.endswith(("embed_sum", "embedding_sum")):
                        value.normal_(0, 0.2)
                    elif name.endswith("cluster_usage"):
                        value.fill_(2)
            signal = torch.randn(1, 1, samples) * 0.15
            codes = encoder.encode(signal, num_quantizers=books).audio_codes
            waveform = decoder(codes).flatten().numpy()
        state = {
            f"{prefix}.{k}": v.contiguous()
            for prefix, module in [("encoder", encoder), ("decoder", decoder)]
            for k, v in module.state_dict().items()
        }
        save_file(state, directory / "audio_tokenizer/model.safetensors")
        (directory / "config.json").write_text(json.dumps(c))
        (directory / "audio_tokenizer/config.json").write_text(json.dumps(a))
        signal.numpy().tofile(directory / "input.f32")
        codes[0].T.contiguous().numpy().astype(np.int32).tofile(
            directory / "expected.codes"
        )
        waveform.tofile(directory / "expected.audio")
        cases.append(directory.name)
    (output / "cases.json").write_text(json.dumps(cases))


def verify(binary, output):
    for name in json.loads((output / "cases.json").read_text()):
        directory = output / name
        subprocess.run(
            [
                str(binary),
                str(directory),
                str(directory / "input.f32"),
                str(directory / "native"),
            ],
            check=True,
        )
        expected = np.fromfile(directory / "expected.codes", np.int32)
        actual = np.fromfile(directory / "native.codes", np.int32)
        np.testing.assert_array_equal(actual, expected)
        expected_audio = np.fromfile(directory / "expected.audio", np.float32)
        actual_audio = np.clip(
            np.fromfile(directory / "native.audio", np.float32), -1, 1
        )
        np.testing.assert_allclose(actual_audio, expected_audio, rtol=2e-3, atol=2e-5)
        print(
            name,
            "codes identical; waveform max_abs",
            float(np.max(np.abs(actual_audio - expected_audio))),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "verify", "all"), default="all")
    args = parser.parse_args()
    if args.stage != "verify":
        if args.model is None:
            parser.error("--model is required for fixture preparation")
        prepare(args.model, args.output)
    if args.stage != "prepare":
        if args.binary is None:
            parser.error("--binary is required for native verification")
        verify(args.binary.resolve(), args.output)
