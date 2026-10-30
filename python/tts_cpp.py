"""TVM FFI bindings to the native C++/CUDA engine; no Python model execution."""

from array import array
from pathlib import Path
import threading
import uuid

import request_schema
from model_source import resolve_model_dir
from native_library import load_library
from native_types import Options
from tts_logging import check_not_in_callback, configure_logging  # noqa: F401 -- public API


class Reference:
    def __init__(self, module, handle):
        self._module, self._handle = module, handle

    def _native_handle(self):
        handle = self._handle
        if handle is None:
            raise RuntimeError("reference is closed")
        return handle

    @property
    def frames(self):
        return self._module.reference_frames(self._native_handle())

    def codes(self):
        return list(self._module.reference_codes(self._native_handle()))

    def close(self):
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
        runtime_memory_mb=0,
        memory_limit_mb=0,
    ):
        check_not_in_callback()
        if quantization not in (None, "fp8", "int8"):
            raise ValueError("quantization must be fp8 or int8")
        if type(runtime_memory_mb) is not int or not 0 <= runtime_memory_mb < 2**43:
            raise ValueError("runtime_memory_mb must be a nonnegative integer")
        if type(memory_limit_mb) is not int or not 0 <= memory_limit_mb < 2**43:
            raise ValueError("memory_limit_mb must be a nonnegative integer")
        self.quantization = quantization
        self._module = load_library(library)
        self.model = resolve_model_dir(model, revision)
        self._callbacks = threading.local()
        self.max_batch, self.context = max_batch, context
        self._handle = self._module.create(
            str(self.model),
            device,
            max_batch,
            context,
            quantization or "",
            runtime_memory_mb * 1024 * 1024,
            memory_limit_mb * 1024 * 1024,
        )
        self._format = dict(self._module.audio_format(self._handle))

    @property
    def audio_format(self):
        return dict(self._format)

    @property
    def version(self):
        return self._module.version()

    def _native_handle(self):
        check_not_in_callback()
        handle = self._handle
        if handle is None:
            raise RuntimeError("engine is closed")
        if getattr(self._callbacks, "active", False):
            raise RuntimeError("cannot reenter engine from an audio callback")
        return handle

    def run_async(self):
        """Start native scheduling and execution; construction only loads resources."""
        self._module.run_async(self._native_handle())

    def close(self):
        check_not_in_callback()
        if getattr(self._callbacks, "active", False):
            raise RuntimeError("cannot reenter engine from an audio callback")
        handle = self._handle
        if handle is not None:
            self._module.close(handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def default_options(self):
        return self._module.default_options()

    def memory_stats(self):
        """Pool bytes between requests; excludes CUDA driver/library memory."""
        return dict(self._module.memory_stats(self._native_handle()))

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
        handle = self._module.prepare_reference(
            self._native_handle(), data, transcript, sample_rate or 0, channels
        )
        return Reference(self._module, handle)

    def reference_from_codes(self, codes, transcript):
        if (
            not isinstance(transcript, str)
            or not transcript.strip()
            or "\0" in transcript
            or len(transcript.encode()) > 65536
            or not codes
            or len(codes) % self._format["codebooks"]
        ):
            raise ValueError(
                "reference requires transcript and complete code frames for this model"
            )
        if any(
            not isinstance(c, int) or not 0 <= c < self._format["codebook_size"]
            for c in codes
        ):
            raise ValueError(
                "reference codes must be integers within the model codebook"
            )
        return Reference(
            self._module,
            self._module.reference_from_codes(
                list(codes),
                transcript,
                self._format["codebooks"],
                self._format["codebook_size"],
            ),
        )

    def _snapshot_generation(self, requests, options):
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
        native = []
        for request in requests:
            reference = request.get("reference")
            if reference is not None and not isinstance(reference, Reference):
                raise ValueError("reference must be a prepared Reference")
            # Local ObjectRefs retain native storage across a concurrent close.
            handle = reference._handle if reference is not None else None
            if reference is not None and handle is None:
                raise ValueError("reference is closed")
            native.append(
                dict(
                    text=request["text"],
                    instruction=request.get("instruction", ""),
                    speaker=request.get("speaker", "S0") or "",
                    reference=handle,
                )
            )
        return native, options

    def configure_serving(self, request_limit=32, output_buffer_bytes=1 << 20):
        """Set native admission and output buffer limits for future requests."""
        self._module.configure_serving(
            self._native_handle(), request_limit, output_buffer_bytes
        )

    def add_request(self, request):
        """Enqueue {request_id, inputs, options} without waiting for native scheduling."""
        check_not_in_callback()
        if not isinstance(request, dict):
            raise TypeError("request must be a dictionary")
        request_id = request.get("request_id")
        if not isinstance(request_id, str) or not request_id or "\0" in request_id:
            raise ValueError("request_id must be a nonempty string without NUL")
        handle = self._native_handle()
        native, options = self._snapshot_generation(
            request.get("inputs"), request.get("options")
        )
        self._module.add_request(handle, request_id, native, options)

    def read_result(self, request_id, timeout=None):
        """Read by ID; terminal output/errors consume it. Duplicate errors preserve it."""
        check_not_in_callback()
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be nonnegative or None")
        handle = self._native_handle()
        frame = dict(
            self._module.read_result(
                handle, request_id, -1 if timeout is None else int(timeout * 1000)
            )
        )
        frame["chunks"] = [(int(c["index"]), bytes(c["pcm"])) for c in frame["chunks"]]
        return frame

    def cancel_request(self, request_id):
        """Post cancellation and return immediately; unknown IDs are ignored."""
        return self._module.cancel_request(self._native_handle(), request_id)

    def generate(self, requests, options=None, deliver=None):
        """Collect PCM or deliver(index, bytes); returning False cancels this request."""
        buffers = [bytearray() for _ in requests]
        output = deliver or (lambda i, pcm: buffers[i].extend(pcm))
        request_id = uuid.uuid4().hex
        self.add_request(dict(request_id=request_id, inputs=requests, options=options))
        try:
            while True:
                frame = self.read_result(request_id)
                stats = {
                    k: v
                    for k, v in frame.items()
                    if k not in ("request_id", "chunks", "ready", "done", "batch_size")
                }
                for index, pcm in frame["chunks"]:
                    self._callbacks.active = True
                    try:
                        keep = output(index, pcm) is not False
                    finally:
                        self._callbacks.active = False
                    if not keep:
                        stats["cancelled"] = True
                        return (
                            stats if deliver else ([bytes(b) for b in buffers], stats)
                        )
                if frame["done"]:
                    break
        finally:
            self.cancel_request(request_id)
        return stats if deliver else ([bytes(b) for b in buffers], stats)

    def warmup(self, batch=1, **options):
        return self.generate(
            [{"text": "你好。"} for _ in range(batch)], dict(max_frames=2, **options)
        )[1]
