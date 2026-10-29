#!/usr/bin/env python3
"""Audit nsys SQLite steady-frame intervals for forbidden materializations."""

import argparse
import json
import re
import sqlite3

parser = argparse.ArgumentParser()
parser.add_argument("trace")
args = parser.parse_args()
db = sqlite3.connect(args.trace)
strings = dict(db.execute("SELECT id,value FROM StringIds"))
frames = []
requests = []
all_frames = []
encoders = []
for start, end, text, text_id in db.execute(
    "SELECT start,end,text,textId FROM NVTX_EVENTS WHERE end IS NOT NULL"
):
    if (text or strings.get(text_id)) == "tts.frame.steady":
        frames.append((start, end))
    if (text or strings.get(text_id)) in ("tts.frame.steady", "tts.frame.setup"):
        all_frames.append((start, end))
    if (text or strings.get(text_id)) == "tts.reference_encoder":
        encoders.append((start, end))
    if (text or strings.get(text_id)) == "tts.request":
        requests.append((start, end))
if not frames:
    raise SystemExit("No steady-frame NVTX ranges; capture with --trace=cuda,nvtx")


def steady(t):
    return any(a <= t <= b for a, b in frames)


totals = {}
for start, end, name in db.execute(
    "SELECT start,end,demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL"
):
    if steady(start):
        key = strings[name]
        row = totals.setdefault(key, {"count": 0, "total_us": 0})
        row["count"] += 1
        row["total_us"] += (end - start) / 1000
for name, row in totals.items():
    row["average_us"] = row["total_us"] / row["count"]
if sum(r["count"] for r in totals.values()) < 100 * len(frames):
    raise SystemExit(
        "No steady-frame kernels. Use nsys --cuda-graph-trace=node; graph-only traces cannot establish this contract."
    )
memcopies = []
for start, kind, size in db.execute(
    "SELECT start,copyKind,bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY"
):
    if steady(start):
        memcopies.append({"kind": kind, "bytes": size})
# CUPTI activity memcpy kinds: HtoD=1, DtoH=2, DtoD=8.
bad_copies = [x for x in memcopies if x["kind"] != 2]
forbidden = [
    n
    for n in totals
    if re.search(
        r"cast_kernel|copy_kernel|contiguous|conv_pack|transpose_kernel|gather_last|im2col|history_kernel",
        n,
        re.IGNORECASE,
    )
]
cuda_apis = []
for start, name in db.execute("SELECT start,nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
    if steady(start) and re.search(
        r"cuda(Malloc|Free|MemPool(Create|Destroy|TrimTo)|Memcpy.*Sync|StreamBeginCapture|GraphInstantiate)",
        strings[name],
    ):
        cuda_apis.append(strings[name])
result = {
    "steady_frames": len(frames),
    "forbidden_kernels": forbidden,
    "unexpected_memcopies": bad_copies,
    "allocations_or_captures": sorted(set(cuda_apis)),
    "memcpy_count": len(memcopies),
    "kernels": dict(sorted(totals.items(), key=lambda x: -x[1]["total_us"])),
}
# For a repeated fixed-shape invocation, every request after the first must
# reuse the pool and graphs. H2D prompt/seed/length metadata remains explicit.
warm = sorted(requests)[1:]


def warm_request(t):
    return any(a <= t <= b for a, b in warm)


warm_bad_kernels = []
for start, name in db.execute(
    "SELECT start,demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL"
):
    if warm_request(start) and re.search(
        r"cast_kernel|copy_kernel|contiguous|conv_pack|transpose_kernel|gather_last|im2col|history_kernel",
        strings[name],
        re.IGNORECASE,
    ):
        warm_bad_kernels.append(strings[name])
warm_bad_api = []
for start, name in db.execute("SELECT start,nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
    if warm_request(start) and re.search(
        r"cuda(Malloc|Free|MemPool(Create|Destroy|TrimTo)|HostAlloc|StreamBeginCapture|GraphInstantiate)",
        strings[name],
    ):
        warm_bad_api.append(strings[name])
warm_copies = {}
for start, kind, size in db.execute(
    "SELECT start,copyKind,bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY"
):
    if warm_request(start):
        item = warm_copies.setdefault(kind, {"count": 0, "bytes": 0})
        item["count"] += 1
        item["bytes"] += size
result["warm_requests"] = {
    "count": len(warm),
    "forbidden_kernels": sorted(set(warm_bad_kernels)),
    "allocations_or_captures": sorted(set(warm_bad_api)),
    "copies_by_kind": warm_copies,
}
copies = list(db.execute("SELECT start,copyKind,bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY"))
extra_frame_copies = []
for a, b in all_frames:
    transfers = [(kind, size) for start, kind, size in copies if a <= start <= b]
    # One EOS read, and one PCM read unless all lanes ended. Debug downloads
    # are not allowed in performance traces, including first/setup frames.
    if len(transfers) > 2 or any(kind != 2 for kind, size in transfers):
        extra_frame_copies.append(transfers)
result["unexpected_frame_transfers"] = extra_frame_copies
encoder_bad = []
for a, b in encoders:
    for start, name in db.execute(
        "SELECT start,demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL"
    ):
        if a <= start <= b and re.search(
            r"cast_kernel|copy_kernel|contiguous|conv_pack|transpose_kernel|im2col",
            strings[name],
            re.IGNORECASE,
        ):
            encoder_bad.append(strings[name])
    for start, name in db.execute(
        "SELECT start,nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME"
    ):
        if a <= start <= b and re.search(r"cuda(Malloc|Free|MemPool(Create|Destroy|TrimTo)|HostAlloc)", strings[name]):
            encoder_bad.append(strings[name])
    for start, kind, size in copies:
        if a <= start <= b and kind not in (1, 2):
            encoder_bad.append("memcpy-kind-" + str(kind))
result["reference_encoding"] = {
    "count": len(encoders),
    "forbidden_operations": sorted(set(encoder_bad)),
}
print(json.dumps(result, indent=2))
if (
    forbidden
    or encoder_bad
    or bad_copies
    or cuda_apis
    or warm_bad_kernels
    or warm_bad_api
    or extra_frame_copies
    or any(k not in (1, 2) for k in warm_copies)
):
    raise SystemExit("execution trace audit failed")
