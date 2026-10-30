"""Client metric and streaming transport checks; no native library or GPU needed."""

import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from client_benchmark import distribution, main, run_case


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with self.server.lock:
            self.server.payloads.append(body)
        if self.path == "/error":
            self.send_response(503)
            self.send_header("Content-Length", "4")
            self.end_headers()
            self.wfile.write(b"full")
            return
        if self.server.barrier:
            self.server.barrier.wait(timeout=5)
        self.send_response(200)
        self.send_header("Content-Type", "audio/pcm;rate=1000;channels=1;format=s16le")
        self.send_header("X-Sample-Rate", "1000")
        self.send_header("X-Sample-Format", "s16le")
        self.send_header("X-Samples-Per-Frame", "3")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def chunk(data):
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        if self.path != "/empty":
            # One PCM frame split across HTTP chunks, two more coalesced below.
            chunk(b"\0")
            chunk(b"\0" * 5)
            if self.path == "/broken":
                self.wfile.write(b"6\r\n\0")
                self.wfile.flush()
                self.close_connection = True
                return
            if self.path != "/single":
                time.sleep(0.08)
                chunk(b"\0" * 12)
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


class ClientBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.lock = threading.Lock()
        self.server.payloads = []
        self.server.barrier = None
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_independent_concurrent_streams_and_frame_boundaries(self):
        self.server.barrier = threading.Barrier(3)
        case = run_case(self.url, [dict(text="Hello.")], 3, 6, 5)
        summary = case["summary"]
        self.assertEqual(summary["failed"], 0)
        self.assertEqual(summary["peak_inflight_requests"], 3)
        self.assertEqual(summary["inter_frame_ms"]["count"], 12)
        for result in case["results"]:
            self.assertEqual(result["frames"], 3)
            self.assertAlmostEqual(result["audio_seconds"], 0.009)
            self.assertGreater(result["frame_times_ms"][1] - result["ttft_ms"], 30)
            self.assertAlmostEqual(
                result["tpot_ms"],
                (result["frame_times_ms"][-1] - result["ttft_ms"]) / 2,
            )
        self.assertEqual(len({p["request_id"] for p in self.server.payloads}), 6)
        self.assertTrue(
            all(p["stream"] and "requests" not in p for p in self.server.payloads)
        )

    def test_http_and_truncated_stream_errors_are_excluded(self):
        for path in ("/error", "/broken"):
            case = run_case(self.url + path, [dict(text="Hello.")], 1, 1, 5)
            self.assertEqual(case["summary"]["failed"], 1)
            self.assertEqual(case["summary"]["ttft_ms"]["count"], 0)
            self.assertEqual(case["summary"]["audio_frames_per_s"], 0)
            self.assertIsNotNone(case["results"][0]["error"])
        self.assertEqual(case["results"][0]["frames"], 1)

    def test_empty_and_single_frame_tpot_is_undefined(self):
        for path, frames in (("/empty", 0), ("/single", 1)):
            case = run_case(self.url + path, [dict(text="Hello.")], 1, 1, 5)
            self.assertEqual(case["summary"]["failed"], 0)
            self.assertEqual(case["summary"]["tpot_ms"]["count"], 0)
            self.assertEqual(case["summary"]["empty"], int(frames == 0))
            self.assertEqual(case["results"][0]["frames"], frames)
            self.assertIsNone(case["results"][0]["tpot_ms"])

    def test_cli_jsonl_warmup_and_saved_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "inputs.jsonl"
            output = Path(directory) / "results.json"
            source.write_text(
                '{"text":"Hello.","max_frames":3}\n{"text":"你好。","max_new_tokens":5}\n',
                encoding="utf-8",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                status = main(
                    [
                        "--url",
                        self.url,
                        "--input",
                        str(source),
                        "--output",
                        str(output),
                        "--concurrency",
                        "2",
                        "--requests",
                        "3",
                        "--warmup",
                        "1",
                    ]
                )
            self.assertEqual(status, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(len(self.server.payloads), 5)
        self.assertEqual(len({p["request_id"] for p in self.server.payloads}), 5)
        self.assertTrue(
            any(
                p.get("max_new_tokens") == 5 and "max_frames" not in p
                for p in self.server.payloads
            )
        )
        self.assertEqual(len(report["cases"][0]["results"]), 3)
        self.assertEqual(report["cases"][0]["summary"]["succeeded"], 3)

    def test_percentile_interpolation_and_missing_values(self):
        self.assertEqual(
            distribution([0, None, 100]), dict(count=2, mean=50, p50=50, p95=95, p99=99)
        )
        self.assertIsNone(distribution([])["p50"])


if __name__ == "__main__":
    unittest.main()
