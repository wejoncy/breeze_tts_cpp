"""TVM FFI bindings to the native C++/CUDA engine; no Python model execution."""

from array import array
from contextlib import ExitStack
from pathlib import Path
import threading

import request_schema
from model_source import resolve_model_dir
from native_library import load_library
from native_types import Options
from tts_logging import check_not_in_callback, configure_logging  # noqa: F401 -- public API


class Reference:
    def __init__(self, module, handle):
        self._module, self._handle = module, handle
        self._lock = threading.RLock()

    @property
    def frames(self):
        with self._lock:
            if self._handle is None:
                raise RuntimeError("reference is closed")
            return self._module.reference_frames(self._handle)

    def codes(self):
        with self._lock:
            if self._handle is None:
                raise RuntimeError("reference is closed")
            return list(self._module.reference_codes(self._handle))

    def close(self):
        with self._lock:
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class Engine:
    def __init__(
        self,
        library=None,
        model=".",
        device=0,
        max_batch=8,
        context=2048,
        revision=None,
        quantization=None,
    ):
        check_not_in_callback()
        if quantization not in (None, "fp8", "int8"):
            raise ValueError("quantization must be fp8 or int8")
        self.quantization = quantization
        self._module = load_library(library)
        self.model = resolve_model_dir(model, revision)
        self._lock = threading.RLock()
        self._active = False
        self.max_batch, self.context = max_batch, context
        self._handle = self._module.create(
            str(self.model), device, max_batch, context, quantization or ""
        )

    @property
    def version(self):
        return self._module.version()

    def _check(self):
        check_not_in_callback()
        if self._handle is None:
            raise RuntimeError("engine is closed")
        if self._active:
            raise RuntimeError("cannot reenter engine from an audio callback")

    def close(self):
        check_not_in_callback()
        with self._lock:
            if self._active:
                raise RuntimeError("cannot reenter engine from an audio callback")
            if self._handle is not None:
                self._module.close(self._handle)
                self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def default_options(self):
        return self._module.default_options()

    def memory_stats(self):
        """Pool bytes between requests; excludes CUDA driver/library memory."""
        check_not_in_callback()
        with self._lock:
            self._check()
            return dict(self._module.memory_stats(self._handle))

    def prepare_reference(self, audio, transcript, *, sample_rate=None, channels=1):
        """File/encoded bytes, or interleaved float PCM with sample_rate specified."""
        check_not_in_callback()
        if (
            not isinstance(transcript, str)
            or not transcript.strip()
            or "\0" in transcript
            or len(transcript.encode()) > 65536
        ):
            raise ValueError("reference transcript is required (1..65536 bytes)")
        if sample_rate is None:
            data = (
                Path(audio).read_bytes()
                if isinstance(audio, (str, Path))
                else bytes(audio)
            )
        else:
            if not isinstance(sample_rate, int) or sample_rate <= 0:
                raise ValueError("sample_rate must be a positive integer")
            if not isinstance(channels, int) or channels < 1 or len(audio) % channels:
                raise ValueError("PCM must contain complete interleaved frames")
            data = array("f", audio).tobytes()
        with self._lock:
            self._check()
            handle = self._module.prepare_reference(
                self._handle, data, transcript, sample_rate or 0, channels
            )
            return Reference(self._module, handle)

    def reference_from_codes(self, codes, transcript):
        if (
            not isinstance(transcript, str)
            or not transcript.strip()
            or "\0" in transcript
            or len(transcript.encode()) > 65536
            or not codes
            or len(codes) % 16
        ):
            raise ValueError(
                "reference requires transcript and complete 16-code frames"
            )
        if any(not isinstance(c, int) or not 0 <= c < 2048 for c in codes):
            raise ValueError("reference codes must be integers in 0..2047")
        return Reference(
            self._module, self._module.reference_from_codes(list(codes), transcript)
        )

    def generate(self, requests, options=None, deliver=None):
        """Return (PCM byte arrays, stats), or stream deliver(index, bytes) and return stats.

        Returning False from deliver cancels. Up to 128 requests are split into
        capacity-sized GPU batches, preserving order and per-lane RNG seeds.
        """
        check_not_in_callback()
        if options is None or isinstance(options, dict):
            _, options = request_schema.validate(
                dict(options or {}, requests=requests), True, self.default_options()
            )
        elif isinstance(options, Options):
            # Check options objects as well as dictionaries; snapshot mutable fields.
            options = self._module.copy_options(options)
            request_schema.validate({"requests": requests}, True, options)
        else:
            raise TypeError("options must be a dictionary or native Options")
        buffers = [bytearray() for _ in requests]
        output = deliver or (lambda i, pcm: buffers[i].extend(pcm))
        total = dict(
            prefill_ms=0.0,
            first_audio_ms=0.0,
            total_ms=0.0,
            audio_seconds=0.0,
            frames=0,
            truncated=0,
            cancelled=False,
        )
        with self._lock, ExitStack() as stack:
            self._check()
            refs = {
                id(r["reference"]): r["reference"]
                for r in requests
                if r.get("reference")
            }
            for ref in sorted(refs.values(), key=id):
                if not isinstance(ref, Reference):
                    raise ValueError("reference must be a prepared Reference")
                stack.enter_context(ref._lock)
                if ref._handle is None:
                    raise ValueError("reference is closed")
            # Retain all native handles before callbacks can close a Reference.
            native = [
                dict(
                    text=r["text"],
                    instruction=r.get("instruction", ""),
                    speaker=r.get("speaker", "S0") or "",
                    reference=r["reference"]._handle if r.get("reference") else None,
                )
                for r in requests
            ]
            self._active = True
            try:
                for start in range(0, len(native), self.max_batch):
                    local = self._module.copy_options(options)
                    local.seed = (options.seed + start * 0xD1342543DE82EF95) % 2**64
                    failures = []

                    def callback(index, pcm):
                        try:
                            return output(start + index, pcm) is not False
                        except BaseException as exc:
                            failures.append(exc)
                            return False

                    stats = self._module.generate(
                        self._handle,
                        native[start : start + self.max_batch],
                        local,
                        callback,
                    )
                    if failures:
                        raise failures[0]
                    for key in ("prefill_ms", "total_ms", "audio_seconds", "frames"):
                        total[key] += stats[key]
                    if start == 0:
                        total["first_audio_ms"] = stats["first_audio_ms"]
                    total["truncated"] |= stats["truncated"]
                    if stats["cancelled"]:
                        total["cancelled"] = True
                        break
            finally:
                self._active = False
        return total if deliver else ([bytes(b) for b in buffers], total)

    def warmup(self, batch=1, **options):
        return self.generate(
            [{"text": "你好。"} for _ in range(batch)], dict(max_frames=2, **options)
        )[1]
