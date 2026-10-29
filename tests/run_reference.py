#!/usr/bin/env python3
"""Produce native component dumps and compare against the original Python model.

Requires the development reference environment, not part of the native runtime.
The packed case also repeats codec codes to exercise the 72-frame history window.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--python", default=sys.executable)
parser.add_argument("--output", type=Path)
args = parser.parse_args()
project = Path(__file__).resolve().parents[1]
output = (args.output or project / "artifacts/reference").resolve()
model = args.model.resolve()
binary = str(project / "build/tts")
for name, text, batch in [
    ("short", "你好，欢迎使用语音合成。", 1),
    (
        "packed",
        "你好，欢迎使用语音合成。我们正在检查批量文本编码和连续语音输出是否正确。" * 6,
        2,
    ),
]:
    folder = output / name
    folder.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        TTS_DUMP_PREFIX=str(folder / "native"), TTS_TOKENS_OUT=str(folder / "codes.i32")
    )
    inputs = ["--text", text]
    if batch > 1:
        requests = folder / "requests.jsonl"
        requests.write_text(
            "\n".join(
                json.dumps({"text": t}, ensure_ascii=False) for t in (text, "Hello.")
            )
            + "\n"
        )
        inputs = ["--batch", str(requests)]
    with (folder / "native.log").open("w") as log:
        subprocess.run(
            [
                binary,
                "--model",
                str(model),
                *inputs,
                "--max-frames",
                "8",
                "--temperature",
                "0",
                "--output",
                str(folder / "speech.wav"),
            ],
            env=env,
            stdout=log,
            stderr=log,
            check=True,
        )
    codes = (folder / "codes.i32").read_bytes()
    if batch == 2:
        codes *= 12  # 96 codec frames; only the first frame is used for depth oracle.
        (folder / "codes.i32").write_bytes(codes)
    lane = b"".join(codes[i : i + 64] for i in range(0, len(codes), batch * 64))
    (folder / "lane0.i32").write_bytes(lane)
    with (folder / "codec.log").open("w") as log:
        subprocess.run(
            [
                binary,
                "--model",
                str(model),
                "--decode-codes",
                str(folder / "lane0.i32"),
                "--output",
                str(folder / "codec.f32"),
            ],
            stdout=log,
            stderr=log,
            check=True,
        )
    with (folder / "reference.log").open("w") as log:
        subprocess.run(
            [
                args.python,
                str(project / "tests/validate_reference.py"),
                "--model",
                str(model),
                "--artifacts",
                str(folder),
                "--text",
                text,
                "--batch",
                str(batch),
            ],
            stdout=log,
            stderr=log,
            check=True,
        )
    print(f"{name}: passed; {folder / 'numerical.json'}", flush=True)
