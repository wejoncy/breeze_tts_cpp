#!/usr/bin/env python3
"""Concurrent streaming HTTP benchmark, using only the Python standard library.

Each worker sends independent speech requests and replaces each completed request.
TTFT measures the first complete PCM frame; TPOT measures subsequent audio frames,
not text tokens, RVQ codes, HTTP chunks or TCP packets. Timings include connection,
server queueing, computation and transport, but exclude the client's pending queue.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import http.client
import json
import math
from pathlib import Path
import statistics
import sys
import time
from urllib.parse import urlsplit
import uuid


def measure(url, payload, index, input_index, origin, timeout):
    endpoint = urlsplit(url)
    connection_type = (
        http.client.HTTPSConnection
        if endpoint.scheme == "https"
        else http.client.HTTPConnection
    )
    connection = connection_type(endpoint.hostname, endpoint.port, timeout=timeout)
    target = endpoint.path.rstrip("/") or "/v1/audio/speech"
    if endpoint.query:
        target += "?" + endpoint.query
    payload = dict(payload, stream=True, request_id=uuid.uuid4().hex)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    result = dict(
        index=index,
        input_index=input_index,
        request_id=payload["request_id"],
        http_status=None,
        error=None,
        sample_rate=None,
        samples_per_frame=None,
        frame_times_ms=[],
    )
    started = time.perf_counter()
    result["start_ms"] = (started - origin) * 1000
    try:
        connection.request("POST", target, body, {"Content-Type": "application/json"})
        response = connection.getresponse()
        result["http_status"] = response.status
        if response.status != 200:
            raise ValueError(
                f"HTTP {response.status}: {response.read(4096).decode(errors='replace')}"
            )
        if response.getheader("Content-Type", "").split(";", 1)[0] != "audio/pcm":
            raise ValueError("expected a streaming audio/pcm response")
        if response.getheader("X-Sample-Format") != "s16le":
            raise ValueError("expected mono s16le PCM")
        rate = int(response.getheader("X-Sample-Rate", "0"))
        samples = int(response.getheader("X-Samples-Per-Frame", "0"))
        if rate <= 0 or samples <= 0:
            raise ValueError("missing/invalid audio rate or X-Samples-Per-Frame header")
        result.update(sample_rate=rate, samples_per_frame=samples)
        # http.client removes HTTP framing. Read one model frame regardless of
        # transport fragmentation/coalescing, retaining no waveform in memory.
        while True:
            pcm = response.read(samples * 2)
            if not pcm:
                break
            if len(pcm) != samples * 2:
                raise ValueError("stream ended with an incomplete PCM frame")
            result["frame_times_ms"].append((time.perf_counter() - started) * 1000)
    except (OSError, http.client.HTTPException, ValueError) as error:
        result["error"] = str(error)
    finally:
        connection.close()
    finished = time.perf_counter()
    frames = result["frame_times_ms"]
    gaps = [b - a for a, b in zip(frames, frames[1:])]
    audio_seconds = (
        len(frames) * result["samples_per_frame"] / result["sample_rate"]
        if frames
        else 0
    )
    result.update(
        end_ms=(finished - origin) * 1000,
        e2e_ms=(finished - started) * 1000,
        frames=len(frames),
        audio_seconds=audio_seconds,
        ttft_ms=frames[0] if frames else None,
        tpot_ms=statistics.mean(gaps) if gaps else None,
        rtf=(finished - started) / audio_seconds if audio_seconds else None,
    )
    return result


def distribution(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return dict(count=0, mean=None, p50=None, p95=None, p99=None)

    def percentile(p):
        position = (len(values) - 1) * p
        lower = math.floor(position)
        upper = math.ceil(position)
        return round(
            values[lower] + (values[upper] - values[lower]) * (position - lower), 3
        )

    return dict(
        count=len(values),
        mean=round(statistics.mean(values), 3),
        p50=percentile(0.5),
        p95=percentile(0.95),
        p99=percentile(0.99),
    )


def summarize(results, elapsed, concurrency):
    valid = [r for r in results if r["error"] is None]
    gaps = [
        b - a
        for r in valid
        for a, b in zip(r["frame_times_ms"], r["frame_times_ms"][1:])
    ]
    events = sorted(
        (r[key], delta)
        for r in results
        for key, delta in (("start_ms", 1), ("end_ms", -1))
    )
    active = peak = 0
    for _, delta in events:
        active += delta
        peak = max(peak, active)
    return dict(
        concurrency=concurrency,
        requests=len(results),
        succeeded=len(valid),
        failed=len(results) - len(valid),
        empty=sum(r["frames"] == 0 for r in valid),
        errors=dict(Counter(r["error"] for r in results if r["error"] is not None)),
        elapsed_s=round(elapsed, 3),
        peak_inflight_requests=peak,
        requests_per_s=round(len(valid) / elapsed, 3),
        audio_frames_per_s=round(sum(r["frames"] for r in valid) / elapsed, 3),
        audio_seconds_per_s=round(sum(r["audio_seconds"] for r in valid) / elapsed, 3),
        ttft_ms=distribution(r["ttft_ms"] for r in valid),
        tpot_ms=distribution(r["tpot_ms"] for r in valid),
        inter_frame_ms=distribution(gaps),
        e2e_ms=distribution(r["e2e_ms"] for r in valid),
        rtf=distribution(r["rtf"] for r in valid),
    )


def run_case(url, inputs, concurrency, requests, timeout):
    with ThreadPoolExecutor(max_workers=concurrency) as workers:
        started = time.perf_counter()
        futures = [
            workers.submit(
                measure,
                url,
                inputs[i % len(inputs)],
                i,
                i % len(inputs),
                started,
                timeout,
            )
            for i in range(requests)
        ]
        results = [future.result() for future in as_completed(futures)]
        elapsed = time.perf_counter() - started
    results.sort(key=lambda r: r["index"])
    return dict(summary=summarize(results, elapsed, concurrency), results=results)


def load_inputs(args):
    if args.input:
        inputs = [
            json.loads(line)
            for line in args.input.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        inputs = [
            dict(text=text)
            for text in (
                args.text
                or [
                    "Welcome aboard. We are measuring concurrent streaming speech generation. "
                    "Each request should receive its first audio promptly and continue smoothly."
                ]
            )
        ]
    if not inputs or any(
        not isinstance(p, dict)
        or not isinstance(p.get("text"), str)
        or not p["text"].strip()
        for p in inputs
    ):
        raise ValueError("input must contain nonempty speech JSON objects with text")
    for payload in inputs:
        if "max_new_tokens" not in payload:
            payload.setdefault("max_frames", args.frames)
        payload.setdefault("temperature", 0)
    return inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8088/v1/audio/speech")
    parser.add_argument(
        "--concurrency",
        default="1,4,8",
        help="Comma-separated concurrent request counts",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=32,
        help="Measured requests per concurrency level",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup requests per worker, excluded from metrics",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=64,
        help="Default max_frames; JSONL inputs may override it",
    )
    parser.add_argument(
        "--timeout", type=float, default=120, help="Socket timeout in seconds"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", action="append", help="Repeat for a text corpus")
    source.add_argument(
        "--input",
        type=Path,
        help="JSONL speech payloads, cycled independently across requests",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report with every frame timestamp; no audio is saved",
    )
    args = parser.parse_args(argv)
    try:
        levels = [int(value) for value in args.concurrency.split(",")]
        endpoint = urlsplit(args.url)
        if (
            endpoint.scheme not in ("http", "https")
            or not endpoint.hostname
            or endpoint.username is not None
            or endpoint.fragment
            or endpoint.port == 0
        ):
            raise ValueError(
                "url must be an HTTP(S) endpoint without credentials or fragment"
            )
        if (
            min(levels) < 1
            or args.requests < 1
            or args.frames < 1
            or args.warmup < 0
            or not math.isfinite(args.timeout)
            or args.timeout <= 0
        ):
            raise ValueError("counts and timeout must be positive; warmup may be zero")
        inputs = load_inputs(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    report = dict(
        url=args.url,
        workload="closed_loop",
        warmup_per_worker=args.warmup,
        metrics=dict(
            ttft_ms="Request start (including connect) to first complete PCM frame",
            tpot_ms="Per-request mean interval between complete audio frames; requires at least two",
            inter_frame_ms="All successful requests' observed frame intervals, including stalls",
            timing="Client observations include server queueing and transport; exclude client pending queue",
            throughput="Successful requests/audio only; failed and partial streams are reported separately",
        ),
        cases=[],
    )
    for concurrency in levels:
        if args.warmup:
            warm = run_case(
                args.url, inputs, concurrency, concurrency * args.warmup, args.timeout
            )
            if warm["summary"]["failed"]:
                print(json.dumps(dict(warmup_failed=warm["summary"])), file=sys.stderr)
                return 1
        case = run_case(args.url, inputs, concurrency, args.requests, args.timeout)
        report["cases"].append(case)
        print(json.dumps(case["summary"], ensure_ascii=False), flush=True)
    if args.output:
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return int(any(case["summary"]["failed"] for case in report["cases"]))


if __name__ == "__main__":
    sys.exit(main())
