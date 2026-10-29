"""Compare independent text segments, audio EOS, and all CFG backbone lanes.

Uses native component dumps and the original Python model only as an oracle.
"""

import argparse
from pathlib import Path
import sys
import json
import numpy as np
import torch
from transformers import AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--model", default=".")
p.add_argument("--prefix", required=True)
p.add_argument("--requests", required=True)
p.add_argument("--dual", action="store_true")
p.add_argument("--cfg", type=float, default=1)
a = p.parse_args()
sys.path.insert(0, str(Path(a.model) / "breeze-tts"))
from models.breeze import BreezeForConditionalGeneration
from breeze_infer.templates import get_template, prepare_inputs, select_template_name
from qwen_tts import Qwen3TTSTokenizer

requests = [
    json.loads(line)
    for line in Path(a.requests).read_text().splitlines()
    if line.strip()
]
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
model = (
    BreezeForConditionalGeneration.from_pretrained(
        a.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    .cuda()
    .eval()
)
tok = AutoTokenizer.from_pretrained(a.model, fix_mistral_regex=False)
audio = Qwen3TTSTokenizer.from_pretrained(
    str(Path(a.model) / "audio_tokenizer"),
    device_map="cuda",
    dtype=torch.float32,
    attn_implementation="sdpa",
)
# Condition branches are evaluated individually with original templates so every
# branch's reference/text isolation is checked, including the independent BOS.
branches = []
if a.dual:
    for mode in ("uncond", "ref", "ins"):
        for req in requests:
            req = dict(req)
            if mode != "ref":
                req.pop("ref_audio_path", None)
                req.pop("ref_text", None)
            if mode != "ins":
                req.pop("instruction", None)
            branches.append(req)
else:
    branches = list(requests)
    if a.cfg != 1:
        branches += [dict(r, instruction="") for r in requests]
expected_prompts = []
expected_backbone = []
lengths = []
with torch.inference_mode():
    for req in branches:
        inputs = prepare_inputs(
            tok,
            audio,
            model,
            [req],
            get_template(select_template_name(req)),
            guidance_scale=1.0,
            guidance_scale_ref=None,
            guidance_scale_ins=None,
        )
        merged = model._merge_input_ids_with_input_values(**inputs)
        print("merged keys", merged.keys(), flush=True)
        prompt = merged["inputs_embeds"]
        lengths.append(prompt.shape[1])
        expected_prompts.append(prompt.squeeze(0))
        # Feed native prompt into the backbone to isolate numerical backbone error.
    all_native = (
        torch.from_numpy(
            np.fromfile(a.prefix + ".reference_prompt.bf16", dtype=np.uint16)
        )
        .view(torch.bfloat16)
        .reshape(-1, 2048)
        .cuda()
    )
    row = 0
    for n in lengths:
        expected_backbone.append(
            model.backbone_model(
                inputs_embeds=all_native[row : row + n].unsqueeze(0), use_cache=False
            ).last_hidden_state[0, -1]
        )
        row += n
    expected_prompt = torch.cat(expected_prompts)
    expected_backbone = torch.stack(expected_backbone)
    native_backbone = (
        torch.from_numpy(
            np.fromfile(a.prefix + ".reference_backbone.bf16", dtype=np.uint16)
        )
        .view(torch.bfloat16)
        .reshape(-1, 2048)
        .cuda()
    )

    def compare(x, y):
        x = x.float().flatten()
        y = y.float().flatten()
        return dict(
            relative_l2=((x - y).norm() / x.norm()).item(),
            cosine=torch.nn.functional.cosine_similarity(x, y, dim=0).item(),
            max_abs=(x - y).abs().max().item(),
        )

    result = {
        "prompt": compare(expected_prompt, all_native),
        "backbone": compare(expected_backbone, native_backbone),
        "lengths": lengths,
    }
    print(json.dumps(result, indent=2))
    Path(a.prefix + ".validation.json").write_text(json.dumps(result, indent=2))
    assert (
        result["prompt"]["relative_l2"] < 0.04
        and result["backbone"]["relative_l2"] < 0.04
    )
