#!/usr/bin/env python3
"""Compare BF16/FP8 prefill before generated token histories can diverge.

Uses existing explicit diagnostic dumps; these runs are not performance runs.
Reference codes are encoded once with BF16 and imported unchanged into FP8.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from tts_cpp import Engine

STAGES = ("text_segments", "reference_backbone", "backbone_logits")


def read_bf16(path):
    return (
        (np.fromfile(path, dtype="<u2").astype(np.uint32) << 16)
        .view(np.float32)
        .astype(np.float64)
    )


def metrics(reference, actual):
    if (
        reference.shape != actual.shape
        or not np.isfinite(actual).all()
        or not np.isfinite(reference).all()
    ):
        raise ValueError("tensor shape/nonfinite mismatch")
    delta = actual - reference
    norm = np.linalg.norm(reference)
    return {
        "elements": reference.size,
        "relative_l2": float(np.linalg.norm(delta) / max(norm, 1e-30)),
        "mae": float(np.abs(delta).mean()),
        "max_absolute": float(np.abs(delta).max()),
        "cosine": float(
            np.dot(reference, actual) / max(norm * np.linalg.norm(actual), 1e-30)
        ),
    }


def log_probability(logits):
    shifted = logits - logits.max()
    return shifted - np.log(np.exp(shifted).sum())


def run(args):
    cases = json.loads(args.manifest.read_text())["cases"]
    args.output.mkdir(parents=True, exist_ok=True)
    codes = {}
    saved = os.environ.get("TTS_DUMP_PREFIX")
    try:
        os.environ.pop("TTS_DUMP_PREFIX", None)
        for precision in ("bf16", args.quantization):
            with Engine(
                model=args.model,
                quantization=None if precision == "bf16" else precision,
                max_batch=1,
            ) as engine:
                for case in cases:
                    name = case["case"]
                    if precision == "bf16":
                        with engine.prepare_reference(
                            case["reference"]["wav"], case["reference"]["text"]
                        ) as ref:
                            codes[name] = ref.codes()
                    with engine.reference_from_codes(
                        codes[name], case["reference"]["text"]
                    ) as ref:
                        os.environ["TTS_DUMP_PREFIX"] = str(
                            (args.output / f"{precision}-{name}").resolve()
                        )
                        engine.generate(
                            [{"text": case["expected"], "reference": ref}],
                            {
                                "max_frames": 1,
                                "temperature": 0,
                                "use_cuda_graph": False,
                            },
                        )
                        os.environ.pop("TTS_DUMP_PREFIX", None)
                    print(precision, name, flush=True)
    finally:
        if saved is None:
            os.environ.pop("TTS_DUMP_PREFIX", None)
        else:
            os.environ["TTS_DUMP_PREFIX"] = saved
    records = []
    for case in cases:
        result = {"case": case["case"], "language": case["language"], "stages": {}}
        for stage in STAGES:
            a, b = [
                read_bf16(args.output / f"{precision}-{case['case']}.{stage}.bf16")
                for precision in ("bf16", args.quantization)
            ]
            if stage == "backbone_logits":
                # Exactly the currently supported Breeze sampler's valid first-code IDs.
                valid = np.r_[np.arange(2048), 2051]
                if a.size <= valid.max():
                    raise ValueError("unexpected Breeze first-code vocabulary")
                a, b = a[valid], b[valid]
                pa, pb = log_probability(a), log_probability(b)
                order = np.argsort(-a, kind="stable")
                result["first_token"] = {
                    "bf16_top1": int(valid[np.argmax(a)]),
                    f"{args.quantization}_top1": int(valid[np.argmax(b)]),
                    "top1_equal": bool(np.argmax(a) == np.argmax(b)),
                    "bf16_top1_margin": float(a[order[0]] - a[order[1]]),
                    f"kl_bf16_to_{args.quantization}_T1": float(
                        np.sum(np.exp(pa) * (pa - pb))
                    ),
                    "total_variation_T1": float(
                        np.abs(np.exp(pa) - np.exp(pb)).sum() / 2
                    ),
                    "top10_overlap": len(
                        set(order[:10]) & set(np.argsort(-b, kind="stable")[:10])
                    )
                    / 10,
                }
            result["stages"][stage] = metrics(a, b)
        records.append(result)
    summary = {}
    for stage in STAGES:
        summary[stage] = {
            metric: {
                "mean": float(np.mean(v)),
                "max": float(np.max(v)),
                "min": float(np.min(v)),
            }
            for metric in ("relative_l2", "mae", "max_absolute", "cosine")
            for v in [[r["stages"][stage][metric] for r in records]]
        }
    summary["first_token"] = {
        "top1_equal": sum(r["first_token"]["top1_equal"] for r in records),
        "cases": len(records),
        **{
            metric: float(np.mean([r["first_token"][metric] for r in records]))
            for metric in (
                f"kl_bf16_to_{args.quantization}_T1",
                "total_variation_T1",
                "top10_overlap",
            )
        },
    }
    report = {
        "quantization": args.quantization,
        "scope": f"{len(cases)} fixed reference-conditioned prompts, batch=1, prefill and first-code logits only; "
        "no generated history. Relative L2/cosine per case; summary is unweighted case mean. "
        "Logit distribution metrics use valid code IDs, temperature=1 and no top-k filtering.",
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "library_sha256": hashlib.sha256(
            (ROOT / "build/libtts.so").read_bytes()
        ).hexdigest(),
        "reference_codes": codes,
        "summary": summary,
        "records": records,
    }
    (args.output / "numeric.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quantization", choices=("fp8", "int8"), default="fp8")
    run(parser.parse_args())
