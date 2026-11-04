"""Shared Python request schema. No engine, HTTP, or device dependencies."""

import math

MAX_INPUTS = 128


class Overloaded(RuntimeError):
    pass


def validate(payload, batch, defaults):
    if not isinstance(payload, dict):
        raise ValueError("JSON object required")
    requests = payload.get("requests") if batch else [payload]
    if not isinstance(requests, list) or not 1 <= len(requests) <= MAX_INPUTS:
        raise ValueError(f"requests must contain 1..{MAX_INPUTS} entries")
    for r in requests:
        if not isinstance(r, dict):
            raise ValueError("each request must be an object")
        r.setdefault("language", payload.get("language", ""))
        r.setdefault("text_split_method", payload.get("text_split_method", ""))
        r.setdefault("fragment_interval", payload.get("fragment_interval", 0.0))
        if r["text_split_method"] not in ("", "cut0", "cut1", "cut2", "cut3", "cut4", "cut5"):
            raise ValueError("invalid text_split_method")
        pause = r["fragment_interval"]
        if isinstance(pause, bool) or not isinstance(pause, (float, int)) or not math.isfinite(pause) or not 0 <= pause <= 10:
            raise ValueError("invalid fragment_interval")
        for field, limit in [("text", 65536), ("instruction", 16384), ("speaker", 256), ("language", 32)]:
            val = r.get(field, "")
            if field == "speaker" and val is None:
                continue
            if not isinstance(val, str) or "\0" in val or len(val.encode()) > limit:
                raise ValueError("invalid " + field)
        if not r.get("text", "").strip():
            raise ValueError("text is required")
    opts = defaults
    aliases = {
        "max_new_tokens": "max_frames",
        "streaming_window_mode": "synthesis_mode",
        "gpt_streaming_lookahead_seconds": "semantic_lookahead_seconds",
        **{
            "depth_decoder_" + k: "depth_" + k
            for k in ("temperature", "top_k", "top_p", "do_sample")
        },
    }
    values = {aliases.get(k, k): v for k, v in payload.items()}
    # Preserve the existing shared CLI/JSON temperature/top-k convenience. Explicit
    # depth values always override it; all controls remain independently available.
    for key in ("temperature", "top_k", "top_p", "do_sample"):
        if key in values:
            values.setdefault("depth_" + key, values[key])
    fields = type(opts).__tvm_ffi_type_info__.fields
    for field in fields:
        name = field.name
        if name not in values:
            continue
        value = values[name]
        kind = type(getattr(opts, name))
        flag = kind is bool
        if issubclass(kind, str):
            if name != "synthesis_mode" or value not in ("offline", "fixed", "hybrid"):
                raise ValueError("invalid " + name)
            setattr(opts, name, value)
            continue
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or (isinstance(value, bool) and not flag)
        ):
            raise ValueError("invalid " + name)
        if kind is not float:
            if not isinstance(value, int):
                raise ValueError(name + " must be integer")
            if not (
                0 <= value < 2**64 if name == "seed" else -(2**31) <= value < 2**31
            ):
                raise ValueError(name + " outside integer range")
        if flag and value not in (0, 1):
            raise ValueError("invalid " + name)
        setattr(opts, name, bool(value) if flag else value)
    if ("cfg_scale_ref" in values) != ("cfg_scale_ins" in values):
        raise ValueError("both cfg_scale_ref and cfg_scale_ins are required")
    if "cfg_scale_ref" in values:
        opts.dual_cfg = True
    if opts.max_frames < 1 or opts.cfg_scale < 0 or opts.repetition_penalty <= 0:
        raise ValueError("generation option outside supported range")
    if not 0.1 <= opts.speed_factor <= 4 or opts.sample_steps < 0 or opts.cfg_rate < -1:
        raise ValueError("invalid synthesis controls")
    for prefix in ("", "depth_"):
        if (
            getattr(opts, prefix + "temperature") < 0
            or getattr(opts, prefix + "top_k") < 0
            or not 0 < getattr(opts, prefix + "top_p") <= 1
            or getattr(opts, prefix + "do_sample") not in (0, 1)
        ):
            raise ValueError("invalid sampling configuration")
    for field in fields:
        value = getattr(opts, field.name)
        if not isinstance(value, str) and not math.isfinite(value):
            raise ValueError("nonfinite " + field.name)
    if opts.use_cuda_graph not in (0, 1) or opts.dual_cfg not in (0, 1):
        raise ValueError("invalid execution flag")
    if "stream" in payload and not isinstance(payload["stream"], bool):
        raise ValueError("stream must be boolean")
    return requests, opts
