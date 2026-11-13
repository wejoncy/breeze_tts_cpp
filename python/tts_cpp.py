"""TVM FFI bindings to the native C++/CUDA engine; no Python model execution."""

from array import array
from pathlib import Path
import uuid

import request_schema
from model_source import resolve_model_dir
from native_library import load_library
from native_types import Options
from tts_logging import configure_logging  # noqa: F401 -- public API

QUANTIZATIONS = ("int8", "fp8", "group_int8", "group_fp8")


class _OutputQueue:
    def __init__(self, owner):
        self.owner = owner

    def get(self, timeout=None):
        """Pop the next result, regardless of request ID."""
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be nonnegative or None")
        frame = dict(self.owner._module.pop_output(
            self.owner._native_handle(), -1 if timeout is None else int(timeout * 1000)))
        if not frame:
            raise TimeoutError
        frame["chunks"] = [(int(c["index"]), bytes(c["pcm"])) for c in frame["chunks"]]
        return frame


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
        if quantization is not None and quantization not in QUANTIZATIONS:
            raise ValueError("quantization must be " + ", ".join(QUANTIZATIONS))
        if type(runtime_memory_mb) is not int or not 0 <= runtime_memory_mb < 2**43:
            raise ValueError("runtime_memory_mb must be a nonnegative integer")
        if type(memory_limit_mb) is not int or not 0 <= memory_limit_mb < 2**43:
            raise ValueError("memory_limit_mb must be a nonnegative integer")
        self.quantization = quantization
        self._module = load_library(library)
        self.model = resolve_model_dir(model, revision)
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
    def model_info(self):
        info = dict(self._module.model_info(self._native_handle()))
        info["checkpoint_files"] = list(info["checkpoint_files"])
        return info

    @property
    def version(self):
        return self._module.version()

    def _native_handle(self):
        handle = self._handle
        if handle is None:
            raise RuntimeError("engine is closed")
        return handle

    def run_async(self):
        """Start native scheduling and execution; construction only loads resources."""
        self._module.run_async(self._native_handle())

    def close(self):
        handle = self._handle
        if handle is not None:
            self._module.close(handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def default_options(self):
        return self._module.default_options(self._native_handle())

    def prepare_reference(self, audio, transcript, *, sample_rate=None, channels=1, language=""):
        """File/encoded bytes, or interleaved float PCM with sample_rate specified."""
        if not isinstance(language, str) or "\0" in language or len(language) > 32:
            raise ValueError("invalid reference language")
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
            self._native_handle(), data, transcript, sample_rate or 0, channels, language
        )
        return Reference(self._module, handle)

    def reference_from_codes(self, codes, transcript):
        if self._format["codebooks"] <= 0:
            raise ValueError("loaded model does not expose discrete audio codes")
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
                    language=request.get("language", ""),
                    text_split_method=request.get("text_split_method", ""),
                    fragment_interval=request.get("fragment_interval", 0.0),
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

    @property
    def output_queue(self):
        """The shared native audio queue; one application owns its consumption."""
        return _OutputQueue(self)

    def cancel_request(self, request_id):
        """Post cancellation and return immediately; unknown IDs are ignored."""
        return self._module.cancel_request(self._native_handle(), request_id)

    def generate(self, requests, options=None, deliver=None):
        """Collect PCM or deliver(index, bytes); returning False cancels this request."""
        buffers = [bytearray() for _ in requests]
        output = deliver or (lambda i, pcm: buffers[i].extend(pcm))
        request_id = uuid.uuid4().hex
        self.add_request(dict(request_id=request_id, inputs=requests, options=options))
        cancelled = False
        terminal = False
        sink_error = None
        try:
            while True:
                frame = self.output_queue.get()
                if frame["request_id"] != request_id:
                    raise RuntimeError("generate requires exclusive ownership of output_queue")
                terminal = frame["done"]
                if frame.get("error"):
                    raise RuntimeError(frame["error"])
                completion = {"cancelled": frame["cancelled"], "truncated": frame["truncated"]}
                if cancelled:
                    if terminal:
                        completion["cancelled"] = True
                        break
                    continue
                for index, pcm in frame["chunks"]:
                    try:
                        keep = output(index, pcm) is not False
                    except Exception as error:
                        sink_error = error
                        keep = False
                    if not keep:
                        completion["cancelled"] = True
                        cancelled = True
                        if not terminal:
                            self.cancel_request(request_id)
                        break
                if frame["done"]:
                    break
        finally:
            if not terminal:
                self.cancel_request(request_id)
        if sink_error is not None:
            raise sink_error
        return completion if deliver else ([bytes(b) for b in buffers], completion)

    def warmup(self, batch=1, *, text="你好。", reference=None, **options):
        return self.generate(
            [{"text": text, "reference": reference} for _ in range(batch)],
            dict(max_frames=2, **options),
        )[1]
