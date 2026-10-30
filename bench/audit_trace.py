#!/usr/bin/env python3
"""Audit serving NVTX ranges in an nsys SQLite export (requires graph node tracing)."""

import argparse
import json
import re
import sqlite3
from bisect import bisect_right

FORBIDDEN_KERNEL = re.compile(
    r"cast_kernel|copy_kernel|contiguous|conv_pack|transpose_kernel|gather_last|im2col|history_kernel|initialize_.*tuning",
    re.IGNORECASE,
)
RESOURCE_API = re.compile(
    r"(?:cuda|cu)(?:.*Alloc|.*Free|MemPool(?:Create|Destroy|TrimTo)|MemCreate|MemMap|MemUnmap|MemRelease|StreamCreate|EventCreate|StreamBeginCapture|StreamEndCapture|GraphInstantiate)",
    re.IGNORECASE,
)


class Intervals:
    def __init__(self, ranges):
        merged = []
        for start, end in sorted(ranges):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        self.ranges = merged
        self.starts = [a for a, _ in merged]

    def __contains__(self, timestamp):
        index = bisect_right(self.starts, timestamp) - 1
        return index >= 0 and timestamp < self.ranges[index][1]


def audit(db):
    strings = dict(db.execute("SELECT id,value FROM StringIds"))
    forbidden_names = {
        i for i, name in strings.items() if FORBIDDEN_KERNEL.search(name)
    }
    resource_names = {i for i, name in strings.items() if RESOURCE_API.search(name)}
    ranges = {}
    for start, end, text, text_id in db.execute(
        "SELECT start,end,text,textId FROM NVTX_EVENTS WHERE end > start"
    ):
        ranges.setdefault(text or strings.get(text_id), []).append((start, end))
    frames = ranges.get("tts.batch.frame", [])
    decode = Intervals(ranges.get("tts.batch.decode", []))
    encoders = ranges.get("tts.reference_encoder", [])
    if not frames or not decode.ranges:
        raise ValueError("No decode frames; capture with --trace=cuda,nvtx")
    # Includes membership uploads, admission, CPU scheduling and reference encoding.
    serving_ranges = frames + encoders + ranges.get("tts.batch.admission", [])
    serving_ranges += ranges.get("tts.scheduler.prepare", [])
    serving = Intervals(
        [(min(a for a, _ in serving_ranges), max(b for _, b in serving_ranges))]
    )
    totals = {}
    decode_kernels = 0
    forbidden = set()
    for start, end, name in db.execute(
        "SELECT start,end,demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL"
    ):
        if start not in serving:
            continue
        if name in forbidden_names:
            forbidden.add(strings[name])
        if start in decode:
            decode_kernels += 1
        row = totals.setdefault(name, {"count": 0, "total_us": 0})
        row["count"] += 1
        row["total_us"] += (end - start) / 1000
    if not decode_kernels:
        raise ValueError(
            "No decode kernels. Use nsys --cuda-graph-trace=node; graph-only traces cannot establish the kernel contract."
        )
    for row in totals.values():
        row["average_us"] = row["total_us"] / row["count"]
    resource_apis = set()
    tables = {
        r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        if table not in tables:
            continue
        for start, name in db.execute(f"SELECT start,nameId FROM {table}"):
            if start in serving and name in resource_names:
                resource_apis.add(strings[name])
    copies = {}
    unexpected = []
    for start, kind, size in db.execute(
        "SELECT start,copyKind,bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY"
    ):
        if start not in serving:
            continue
        row = copies.setdefault(kind, {"count": 0, "bytes": 0})
        row["count"] += 1
        row["bytes"] += size
        # H2D uploads are explicit prompt/metadata/reference transfers. During
        # decode only D2H EOS is needed; PCM follows codec execution.
        if kind not in (1, 2) or (start in decode and kind != 2):
            unexpected.append({"kind": kind, "bytes": size, "start": start})
    return {
        "frames": len(frames),
        "decode_ranges": len(decode.ranges),
        "reference_encodings": len(encoders),
        "forbidden_kernels": sorted(forbidden),
        "allocations_or_captures": sorted(resource_apis),
        "unexpected_memcopies": unexpected,
        "copies_by_kind": copies,
        "kernels": {
            strings[name]: row
            for name, row in sorted(totals.items(), key=lambda x: -x[1]["total_us"])
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace")
    args = parser.parse_args()
    try:
        with sqlite3.connect(args.trace) as db:
            result = audit(db)
    except (ValueError, sqlite3.Error) as error:
        parser.exit(1, str(error) + "\n")
    print(json.dumps(result, indent=2))
    if any(
        result[key]
        for key in (
            "forbidden_kernels",
            "allocations_or_captures",
            "unexpected_memcopies",
        )
    ):
        parser.exit(1, "execution trace audit failed\n")


if __name__ == "__main__":
    main()
