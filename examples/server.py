#!/usr/bin/env python3
"""Standard-library HTTP frontend. All model execution is in the native library.

POST /v1/audio/speech: JSON {text, instruction?, stream?:true, ...options}
  stream=true: chunked mono signed 16-bit little-endian PCM at the model sample rate.
  stream=false: WAV.
POST /v1/audio/batch: JSON {requests:[{text,instruction?},...], ...options}
  Returns a ZIP of WAVs and metrics.json. One native GPU batch invocation.
"""

import argparse
import io
import json
import logging
import threading
import select
import socket
import wave
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


import sys
import base64
import hashlib
from collections import OrderedDict
from email import policy
from email.parser import BytesParser
from urllib.parse import parse_qs

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from tts_cpp import Engine, configure_logging
from model_source import model_info
from request_schema import validate
from serving import Responses, Overloaded


def wav(pcm, sample_rate):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setparams((1, 2, sample_rate, 0, "NONE", "not compressed"))
        w.writeframes(pcm)
    return buf.getvalue()


def make_handler(engine, responses):
    sample_rate = engine.audio_format["sample_rate"]
    samples_per_frame = engine.audio_format["samples_per_frame"]
    references = OrderedDict()
    metadata = None
    metadata_lock = threading.Lock()
    reference_lock = threading.Lock()

    def prepare(payload, requests, uploaded):
        with reference_lock:
            for request in requests:
                transcript = request.get("ref_text", payload.get("ref_text", ""))
                audio = request.get("ref_audio", payload.get("ref_audio"))
                data = uploaded
                if audio:
                    if not isinstance(audio, str):
                        raise ValueError("JSON ref_audio must be base64 encoded audio")
                    data = base64.b64decode(audio, validate=True)
                if bool(data) != bool(transcript):
                    raise ValueError("ref_audio and ref_text must be provided together")
                if data:
                    key = hashlib.sha256(
                        len(data).to_bytes(8, "little") + data + transcript.encode()
                    ).digest()
                    if key not in references:
                        try:
                            references[key] = engine.prepare_reference(data, transcript)
                        except Overloaded:
                            raise
                        except RuntimeError as exc:
                            raise ValueError(str(exc)) from exc
                    references.move_to_end(key)
                    request["reference"] = references[key]
                    # Drop cache ownership only; requests retain their references.
                    while len(references) > 16:
                        references.popitem(last=False)

    def parse_body(body, content_type):
        uploaded = None
        if content_type.startswith("multipart/form-data"):
            message = BytesParser(policy=policy.default).parsebytes(
                (
                    "Content-Type: " + content_type + "\r\nMIME-Version: 1.0\r\n\r\n"
                ).encode()
                + body
            )
            if not message.is_multipart():
                raise ValueError("invalid multipart body")
            payload = {}
            for part in message.iter_parts():
                name = part.get_param("name", header="content-disposition")
                if (
                    not name
                    or name in payload
                    or (name == "ref_audio" and uploaded is not None)
                ):
                    raise ValueError("invalid or duplicate multipart field")
                data = part.get_payload(decode=True)
                if name == "ref_audio":
                    uploaded = data
                else:
                    payload[name] = data.decode("utf-8")
        elif content_type.startswith("application/x-www-form-urlencoded"):
            fields = parse_qs(
                body.decode(), keep_blank_values=True, strict_parsing=True
            )
            if any(len(v) != 1 for v in fields.values()):
                raise ValueError("duplicate form field")
            payload = {k: v[0] for k, v in fields.items()}
        else:
            return json.loads(body), None, False
        for name in (
            "max_frames",
            "max_new_tokens",
            "seed",
            "top_k",
            "depth_top_k",
            "depth_decoder_top_k",
        ):
            if name in payload:
                payload[name] = int(payload[name])
        for name in (
            "temperature",
            "top_p",
            "cfg_scale",
            "cfg_scale_ref",
            "cfg_scale_ins",
            "repetition_penalty",
            "depth_temperature",
            "depth_top_p",
            "depth_decoder_temperature",
            "depth_decoder_top_p",
        ):
            if name in payload:
                payload[name] = float(payload[name])
        for name in (
            "stream",
            "do_sample",
            "depth_do_sample",
            "depth_decoder_do_sample",
            "use_cuda_graph",
            "dual_cfg",
        ):
            if name in payload:
                if payload[name].lower() not in ("true", "false", "1", "0"):
                    raise ValueError("invalid boolean " + name)
                payload[name] = payload[name].lower() in ("true", "1")
        return payload, uploaded, True

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):
            logging.getLogger("tts.http").info(format, *args)

        def log_error(self, format, *args):
            logging.getLogger("tts.http").error(format, *args)

        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def reply(self, status, data, content_type="application/json", headers=None):
            if not isinstance(data, bytes):
                data = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            nonlocal metadata
            if self.path == "/health":
                self.reply(
                    200,
                    {
                        "status": "ready",
                        "engine": engine.version,
                        "sample_rate": sample_rate,
                        "max_batch": engine.max_batch,
                    },
                )
            elif self.path == "/v1/model":
                with metadata_lock:
                    if metadata is None:
                        metadata = model_info(engine.model)
                        metadata["max_new_tokens"] = min(
                            metadata["max_new_tokens"], engine.context - 1
                        )
                self.reply(200, metadata)
            else:
                self.reply(404, {"error": "not found"})

        def do_POST(self):
            if self.path not in (
                "/v1/audio/speech",
                "/v1/audio/batch",
                "/v1/audio/speech/batch",
            ):
                self.close_connection = True
                self.reply(404, {"error": "not found"})
                return
            try:
                if self.headers.get("Transfer-Encoding"):
                    raise ValueError("request requires Content-Length")
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 64 * 1024 * 1024:
                    raise ValueError("request body must be 1..67108864 bytes")
                body = self.rfile.read(size)
                if len(body) != size:
                    raise ValueError("incomplete request body")
                payload, uploaded, form = parse_body(
                    body, self.headers.get("Content-Type", "")
                )
                if not isinstance(payload, dict):
                    raise ValueError("object required")
                compatible_batch = self.path == "/v1/audio/speech/batch"
                batch = self.path.endswith("/batch")
                if compatible_batch:
                    texts = payload.get("texts")
                    texts = json.loads(texts) if isinstance(texts, str) else texts
                    if (
                        not isinstance(texts, list)
                        or not 1 <= len(texts) <= 128
                        or any(not isinstance(t, str) or not t.strip() for t in texts)
                    ):
                        raise ValueError("texts must contain 1..128 nonempty strings")
                    payload["requests"] = [
                        dict(
                            text=t,
                            instruction=payload.get("instruction", ""),
                            speaker=payload.get("speaker", "S0"),
                            ref_audio=payload.get("ref_audio"),
                        )
                        for t in texts
                    ]
                    frames = payload.get(
                        "max_new_tokens",
                        payload.get("max_frames", engine.default_options().max_frames),
                    )
                    if not isinstance(frames, int):
                        raise ValueError("max_new_tokens must be integer")
                    payload.pop("max_new_tokens", None)
                    payload["max_frames"] = frames
                # Schema checks precede reference preparation and native submission.
                if uploaded:
                    for r in payload.get("requests", [payload]):
                        r["ref_audio"] = True
                requests, opts = validate(payload, batch, engine.default_options())
            except (ValueError, TypeError, UnicodeError, OverflowError) as exc:
                self.close_connection = True
                self.reply(400, {"error": str(exc)})
                return
            started = False
            job = None
            try:
                if uploaded:
                    for r in requests:
                        r.pop("ref_audio", None)
                job = responses.submit(
                    requests,
                    opts,
                    lambda admitted: prepare(payload, admitted, uploaded),
                    request_id=payload.get("request_id"),
                )
                streaming = not batch and payload.get("stream", True)
                buffers = [bytearray() for _ in requests]

                def deliver(index, pcm):
                    nonlocal started
                    if streaming:
                        if not started:
                            self.send_response(200)
                            self.send_header(
                                "Content-Type",
                                f"audio/pcm;rate={sample_rate};channels=1;format=s16le",
                            )
                            self.send_header("Transfer-Encoding", "chunked")
                            self.send_header("Cache-Control", "no-store")
                            self.send_header("X-Sample-Rate", str(sample_rate))
                            self.send_header("X-Sample-Format", "s16le")
                            self.send_header(
                                "X-Samples-Per-Frame", str(samples_per_frame)
                            )
                            self.end_headers()
                            started = True
                        self.wfile.write(f"{len(pcm):x}\r\n".encode() + pcm + b"\r\n")
                        self.wfile.flush()
                    else:
                        buffers[index].extend(pcm)

                while True:
                    try:
                        index, pcm = job.take()
                    except TimeoutError:
                        # Detect disconnects while waiting for admission or the next GPU frame.
                        readable, _, _ = select.select([self.connection], [], [], 0)
                        if readable and not self.connection.recv(1, socket.MSG_PEEK):
                            raise ConnectionResetError("client disconnected")
                        continue
                    except StopIteration:
                        break
                    deliver(index, pcm)
                stats = job.stats
                if streaming:
                    if not started:
                        self.reply(
                            200,
                            b"",
                            "audio/pcm",
                            {
                                "X-Sample-Rate": str(sample_rate),
                                "X-Sample-Format": "s16le",
                                "X-Samples-Per-Frame": str(samples_per_frame),
                            },
                        )
                    else:
                        self.wfile.write(b"0\r\n\r\n")
                        self.wfile.flush()
                elif compatible_batch:
                    self.reply(
                        200,
                        b"".join(buffers),
                        "audio/pcm",
                        {
                            "X-Sample-Rate": str(sample_rate),
                            "X-Sample-Format": "s16le",
                            "X-Segment-Bytes": ",".join(str(len(b)) for b in buffers),
                        },
                    )
                elif batch:
                    result = io.BytesIO()
                    with zipfile.ZipFile(result, "w", zipfile.ZIP_STORED) as z:
                        for i, pcm in enumerate(buffers):
                            z.writestr(f"{i}.wav", wav(pcm, sample_rate))
                        z.writestr("metrics.json", json.dumps(stats))
                    self.reply(200, result.getvalue(), "application/zip")
                else:
                    self.reply(200, wav(buffers[0], sample_rate), "audio/wav")
            except Overloaded as exc:
                self.reply(503, {"error": str(exc)})
            except (TimeoutError, BrokenPipeError, ConnectionResetError):
                self.close_connection = True
            except (ValueError, TypeError) as exc:
                if not started:
                    self.reply(400, {"error": str(exc), "detail": str(exc)})
                self.close_connection = True
            except Exception as exc:  # noqa: BLE001 -- HTTP request error boundary
                self.log_error("inference error: %s", exc)
                if not started:
                    self.reply(422, {"error": str(exc)})
                self.close_connection = True
            finally:
                if job is not None:
                    job.cancel()

    return Handler


class Server(ThreadingHTTPServer):
    def __init__(self, address, engine, **limits):
        self.responses = Responses(engine, **limits)
        try:
            super().__init__(address, make_handler(engine, self.responses))
        except BaseException:
            self.responses.close()
            raise

    def server_close(self):
        self.responses.close()
        super().server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument(
        "--library",
        type=Path,
        help="Native library, build directory, or tts-runtime.json",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-batch", type=int, default=8)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument(
        "--memory-limit-mb",
        type=int,
        default=0,
        help="Engine GPU allocation budget in MiB, including weights; 0 uses available memory",
    )
    parser.add_argument(
        "--runtime-memory-mb",
        type=int,
        default=0,
        help="Fixed eager/KV backing in MiB; 0 sizes from context and batch",
    )
    parser.add_argument("--queue-limit", type=int, default=32)
    parser.add_argument("--stream-buffer-mb", type=int, default=1)
    parser.add_argument("--quantization", choices=("fp8", "int8"))
    parser.add_argument(
        "--log-level",
        choices=("trace", "debug", "info", "warn", "error", "off"),
        default="info",
    )
    args = parser.parse_args()
    severity = {
        "trace": 5,
        "debug": 10,
        "info": 20,
        "warn": 30,
        "error": 40,
        "off": 100,
    }[args.log_level]
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(component)s] engine=%(engine_id)s request=%(request_id)s %(message)s",
            defaults={"component": "http", "engine_id": 0, "request_id": 0},
        )
    )
    logging.basicConfig(level=severity, handlers=[handler])
    configure_logging(
        args.log_level, logger=logging.getLogger("tts.native"), library=args.library
    )
    engine = Engine(
        args.library,
        args.model,
        args.device,
        args.max_batch,
        args.context,
        args.revision,
        quantization=args.quantization,
        runtime_memory_mb=args.runtime_memory_mb,
        memory_limit_mb=args.memory_limit_mb,
    )
    engine.run_async()
    if args.warmup:
        engine.warmup()
    server = Server(
        (args.host, args.port),
        engine,
        queue_limit=args.queue_limit,
        buffer_bytes=args.stream_buffer_mb * 1024 * 1024,
    )
    server.daemon_threads = False
    logging.getLogger("tts.http").info("ready on http://%s:%s", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        engine.close()


if __name__ == "__main__":
    main()
