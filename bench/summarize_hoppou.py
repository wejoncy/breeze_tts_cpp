"""Summarize unprofiled, equal-frame HoppouAI/native runs (discard first run)."""

import argparse
import json
from pathlib import Path
from statistics import median

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("baseline", type=Path)
parser.add_argument("native", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
baseline = json.loads(args.baseline.read_text())
native = json.loads(args.native.read_text())
if baseline["prompt_token_counts"] != native["prompt_token_counts"]:
    raise RuntimeError("prompt work differs")
if baseline["reference_frames"] != native["reference_frames"]:
    raise RuntimeError("reference length differs")


def summarize(case):
    warm = case["observations"][1:]
    if len(warm) < 3:
        raise RuntimeError("at least three warm observations are required")
    duration = median(x["total_ms"] for x in warm)
    seconds = warm[0]["audio_seconds"]
    if any(x["audio_seconds"] != seconds for x in warm):
        raise RuntimeError("output work varies")
    return dict(
        total_ms=duration,
        total_min_ms=min(x["total_ms"] for x in warm),
        total_max_ms=max(x["total_ms"] for x in warm),
        first_audio_ms=median(x["first_audio_ms"][0] for x in warm),
        time_to_320ms_audio=median(x["time_to_320ms_audio"][0] for x in warm),
        first_chunk_samples=warm[0]["first_chunk_samples"][0],
        audio_seconds=seconds,
        audio_seconds_per_second=seconds * 1000 / duration,
        rtf=duration / 1000 / seconds,
    )


cases = []
lookup = {c["specification"]["id"]: c for c in native["cases"]}
if len(lookup) != len(baseline["cases"]):
    raise RuntimeError("case counts differ")
for case in baseline["cases"]:
    spec = case["specification"]
    other = lookup[spec["id"]]
    if spec != other["specification"]:
        raise RuntimeError("case specifications differ")
    b, n = summarize(case), summarize(other)
    if b["audio_seconds"] != n["audio_seconds"]:
        raise RuntimeError("engines generated different amounts of audio")
    cases.append(
        dict(
            id=spec["id"],
            requests=len(spec["requests"]),
            baseline=b,
            native=n,
            speedup=b["total_ms"] / n["total_ms"],
        )
    )
result = dict(
    baseline_backend=baseline["backend"],
    native_backend=native["backend"],
    reference_frames=native["reference_frames"],
    cases=cases,
)
args.output.write_text(json.dumps(result, indent=2) + "\n")
for case in cases:
    print(
        f"{case['id']}: {case['baseline']['total_ms']:.2f} -> "
        f"{case['native']['total_ms']:.2f} ms; {case['speedup']:.2f}x"
    )
