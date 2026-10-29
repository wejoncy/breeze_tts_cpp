"""Development-only numerical oracle; inference/build never import these deps.

Run the CLI with TTS_DUMP_PREFIX and TTS_TOKENS_OUT first, then this script.
Each component receives the native preceding component's output to localize
errors; codec compares stateful one-frame native output to full HF decoding.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=Path, default=Path("."))
parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
parser.add_argument("--text", default="你好，欢迎使用语音合成。")
parser.add_argument("--batch", type=int, default=1)
args = parser.parse_args()
sys.path.insert(0, str(args.model.resolve() / "breeze-tts"))
from models.breeze import BreezeForConditionalGeneration
from qwen_tts import Qwen3TTSTokenizer
from transformers import AutoTokenizer

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
results = {}


def native(name, shape):
    raw = np.fromfile(args.artifacts / ("native." + name + ".bf16"), dtype=np.uint16)
    # Debug heads contain every lane; component checks select lane zero.
    raw = raw[: int(np.prod(shape))]
    return torch.from_numpy(raw).view(torch.bfloat16).reshape(shape).cuda()


def compare(name, expected, actual, tolerance):
    a, b = expected.float().reshape(-1), actual.float().reshape(-1)
    rel = ((a - b).norm() / a.norm().clamp_min(1e-10)).item()
    cosine = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    maximum = (a - b).abs().max().item()
    result = {
        "relative_l2": rel,
        "cosine": cosine,
        "max_abs": maximum,
        "tolerance": tolerance,
        "passed": rel < tolerance,
    }
    results[name] = result
    print(name, json.dumps(result), flush=True)


@torch.inference_mode()
def run():
    model = (
        BreezeForConditionalGeneration.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .cuda()
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, fix_mistral_regex=False)
    ids = tokenizer("[S0]" + args.text, return_tensors="pt").input_ids.cuda()
    length = ids.shape[1]
    encoded = model.text_encoder(input_ids=ids).last_hidden_state
    compare("text_encoder", encoded, native("text", (1, length, 1152)), 0.03)
    projected = model.text_encoder_proj(native("text", (1, length, 1152)))
    compare("text_projection", projected, native("projected", (1, length, 2048)), 0.003)
    backbone_result = model.backbone_model(
        inputs_embeds=native("projected", (1, length, 2048)), use_cache=True
    )
    backbone = backbone_result.last_hidden_state[:, -1:]
    compare("backbone", backbone, native("backbone", (1, 1, 2048)), 0.03)
    logits = model.lm_head(native("backbone", (1, 1, 2048)))
    compare("backbone_head", logits, native("backbone_logits", (1, 1, 2052)), 0.003)
    codes = np.fromfile(args.artifacts / "codes.i32", dtype=np.int32).reshape(
        -1, args.batch, 16
    )[:, 0]
    row = torch.tensor(codes[0], device="cuda", dtype=torch.long)
    if (args.artifacts / "native.backbone_step1.bf16").exists():
        offsets = torch.arange(16, device="cuda") * 2051
        inputs = (
            model.depth_decoder.model.embed_tokens(row + offsets)
            .sum(dim=0)
            .reshape(1, 1, 2048)
        )
        step = model.backbone_model(
            inputs_embeds=inputs,
            past_key_values=backbone_result.past_key_values,
            use_cache=True,
        ).last_hidden_state
        compare("backbone_step1", step, native("backbone_step1", (1, 1, 2048)), 0.04)
    all_ids = torch.cat(
        (torch.zeros(1, device="cuda", dtype=torch.long), row[:15])
    ).view(1, -1)
    depth = model.depth_decoder(
        input_ids=all_ids,
        backbone_last_hidden_state=native("backbone", (1, 2048)),
        use_cache=False,
    ).logits
    for i in range(15):
        compare(
            "depth_" + str(i + 1),
            depth[:, i],
            native("depth_" + str(i + 1), (1, 2051)),
            0.03,
        )
    del model
    torch.cuda.empty_cache()
    audio = Qwen3TTSTokenizer.from_pretrained(
        str(args.model / "audio_tokenizer"), device_map="cuda"
    )
    codec_codes = torch.tensor(codes.T[None], device="cuda", dtype=torch.long)
    wave = audio.model.decoder(codec_codes).reshape(-1)
    native_wave = (
        torch.from_numpy(np.fromfile(args.artifacts / "codec.f32", dtype=np.float32))
        .cuda()
        .clamp(-1, 1)
    )
    compare("codec_stream_vs_full", wave, native_wave, 0.02)
    np.save(args.artifacts / "reference_wave.npy", wave.cpu().numpy())


run()
(args.artifacts / "numerical.json").write_text(json.dumps(results, indent=2) + "\n")
if not all(r["passed"] for r in results.values()):
    raise SystemExit("numerical comparison failed")
