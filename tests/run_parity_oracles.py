#!/usr/bin/env python3
"""Reproduce native encoder and dual-CFG conditioning oracles from a checkpoint.
The --python interpreter needs the original project's development dependencies.
No development framework is used by the native executable or frontend.
"""

import argparse
import json
import math
import os
import struct
import subprocess
import sys
import wave
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--model", type=Path, required=True)
p.add_argument("--python", default=sys.executable)
p.add_argument("--output", type=Path)
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
out = (a.output or root / "artifacts/parity-oracles").resolve()
out.mkdir(parents=True, exist_ok=True)
model = a.model.resolve()
binary = str(root / "build/tts")


def run(cmd, name, env=None):
    with (out / (name + ".log")).open("w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)


text = "你好，欢迎使用语音合成。这段声音由原生引擎生成，支持批量处理和实时流式输出。"
reference = out / "reference.wav"
run(
    [
        binary,
        "--model",
        str(model),
        "--text",
        text,
        "--max-frames",
        "180",
        "--output",
        str(reference),
    ],
    "synthesize-reference",
)
with wave.open(str(reference)) as f:
    raw = f.readframes(f.getnframes())
# Force an odd encoder timestep count (>250) and incomplete convolution strides.
frames = 305531
long = (raw * ((2 * frames + len(raw) - 1) // len(raw)))[: frames * 2]


def wav(path, data, rate=24000, channels=1):
    with wave.open(str(path), "wb") as f:
        f.setparams((channels, 2, rate, 0, "NONE", "not compressed"))
        f.writeframes(data)


wav(out / "long-odd.wav", long)
stereo = b"".join(
    struct.pack(
        "<hh",
        int(3200 * math.sin(2 * math.pi * 440 * i / 44100)),
        int(2000 * math.cos(2 * math.pi * 223 * i / 44100)),
    )
    for i in range(70013)
)
wav(out / "stereo.wav", stereo, 44100, 2)
for name in ("reference", "long-odd", "stereo"):
    prefix = out / name
    audio = out / (name + ".wav")
    codes = out / (name + ".i32")
    run(
        [
            binary,
            "--model",
            str(model),
            "--encode-audio",
            str(audio),
            "--output",
            str(codes),
        ],
        name + "-native",
        dict(os.environ, TTS_ENCODER_DUMP=str(prefix)),
    )
    run(
        [
            a.python,
            str(root / "tests/validate_encoder.py"),
            "--audio",
            str(audio),
            "--prefix",
            str(prefix),
            "--codes",
            str(codes),
            "--model",
            str(model / "audio_tokenizer"),
        ],
        name + "-oracle",
    )
requests = [
    dict(
        text="你好，今天很高兴见到你。",
        instruction="A calm, clear voice.",
        ref_audio_path=str(reference),
        ref_text=text,
    ),
    dict(
        text="Welcome aboard.",
        instruction="A warm voice.",
        ref_audio_path=str(reference),
        ref_text=text,
        speaker="S1",
    ),
]
manifest = out / "dual.jsonl"
manifest.write_text(
    "\n".join(json.dumps(r, ensure_ascii=False) for r in requests) + "\n"
)
run(
    [
        binary,
        "--model",
        str(model),
        "--batch",
        str(manifest),
        "--cfg-scale-ref",
        "2",
        "--cfg-scale-ins",
        "3",
        "--max-frames",
        "4",
        "--temperature",
        "0",
        "--output",
        str(out / "dual"),
    ],
    "dual-native",
    dict(os.environ, TTS_DUMP_PREFIX=str(out / "dual")),
)
run(
    [
        a.python,
        str(root / "tests/validate_conditioning.py"),
        "--model",
        str(model),
        "--prefix",
        str(out / "dual"),
        "--requests",
        str(manifest),
        "--dual",
    ],
    "dual-oracle",
)
print("All native/Python encoder and conditioning comparisons passed:", out)
