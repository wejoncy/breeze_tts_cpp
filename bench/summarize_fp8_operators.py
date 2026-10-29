#!/usr/bin/env python3
"""Render measured operator timings without conflating GEMM and quantization."""

import argparse
import json
from pathlib import Path


def render(rows):
    lines = [
        "# FP8 operator measurements on H100 NVL",
        "",
        "GPU CUDA-event timings in microseconds, captured in CUDA graphs. Four warmups,",
        "median of nine measurements. GEMMs rotate weight banks targeting twice L2",
        "capacity; producers use hot inputs. These are isolated operator timings,",
        "not end-to-end inference latency. Split-K GEMM time includes its reduction.",
        "",
        "GEMM rows start with already-quantized activations. FP8 norm and gated rows",
        "include token amax and quantization; BF16 rows do not need quantization.",
        "`out_with_quant` measures the complete dynamic quantization + output GEMM",
        "sequence. `out_complete` measures the prepared production choice, including",
        "input-prologue fusion where it wins. A dash means added work without a BF16",
        "counterpart, not zero speed.",
        "",
    ]
    for model in dict.fromkeys(r["model"] for r in rows):
        lines += [
            f"## {model}",
            "",
            "| Operator | Token rows | N | K | BF16 us | FP8 us | BF16 / FP8 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in rows:
            if r["model"] != model:
                continue
            bf = f"{r['bf16_us']:.3f}" if r["bf16_us"] else "—"
            speed = f"{r['speedup']:.3f}x" if r["bf16_us"] else "—"
            lines.append(
                f"| {r['operator']} | {r['rows']} | {r['n']} | {r['k'] or '—'} "
                f"| {bf} | {r['fp8_us']:.3f} | {speed} |"
            )
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [json.loads(s) for s in args.input.read_text().splitlines() if s.strip()]
    report = render(rows)
    if args.output:
        args.output.write_text(report)
    else:
        print(report)
