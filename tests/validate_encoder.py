"""Development oracle only; native runtime never imports this module."""

import argparse
from pathlib import Path
import json
import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSTokenizer

p = argparse.ArgumentParser()
p.add_argument("--audio", required=True)
p.add_argument("--prefix", required=True)
p.add_argument("--codes", required=True)
p.add_argument("--model", default="audio_tokenizer")
a = p.parse_args()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
model = Qwen3TTSTokenizer.from_pretrained(
    a.model, device_map="cuda", torch_dtype=torch.float32, attn_implementation="sdpa"
)
captures = {}


def capture(name):
    def hook(m, args, out):
        if hasattr(out, "last_hidden_state"):
            out = out.last_hidden_state
        if isinstance(out, tuple):
            out = out[0]
        captures[name] = out.detach().float().cpu().numpy()

    return hook


enc = model.model.encoder
enc.encoder.register_forward_hook(capture("conv"))
enc.encoder_transformer.register_forward_hook(capture("transformer"))
enc.downsample.register_forward_hook(capture("latent"))
wav, sr = sf.read(a.audio, always_2d=True, dtype="float32")
codes = model.encode(wav.mean(axis=1), sr=sr)["audio_codes"][0].cpu().numpy()
np.save(a.prefix + ".reference_codes.npy", codes)
result = {}
for name, v in captures.items():
    v = v.squeeze(0)
    if name != "transformer":
        v = v.T
    x = np.fromfile(a.prefix + "." + name + ".f32", dtype=np.float32).reshape(v.shape)
    np.save(a.prefix + "." + name + ".reference.npy", v)
    result[name] = {
        "shape": list(v.shape),
        "max_abs": float(np.max(np.abs(x - v))),
        "rmse": float(np.sqrt(np.mean((x - v) ** 2))),
        "cosine": float(
            np.dot(x.ravel(), v.ravel()) / np.linalg.norm(x) / np.linalg.norm(v)
        ),
    }
x = np.fromfile(a.codes, dtype=np.int32).reshape(codes.shape)
result["codes"] = {
    "shape": list(codes.shape),
    "agreement": float(np.mean(x == codes)),
    "codebook_agreement": np.mean(x == codes, axis=0).tolist(),
}
print(json.dumps(result, indent=2))
Path(a.prefix + ".validation.json").write_text(json.dumps(result, indent=2))
assert result["codes"]["agreement"] > 0.99, result
