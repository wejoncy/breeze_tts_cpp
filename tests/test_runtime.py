"""Integration checks against the actual checkpoint through TVM FFI.
Run from project root: python3 tests/test_runtime.py --model models/Breeze-TTS-2
"""

import argparse
import http.client
import io
import json
import sys
import threading
import unittest
import wave
import zipfile
from http.server import ThreadingHTTPServer
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--library", type=Path)
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--quantization", choices=("fp8", "int8"))
args, remaining = parser.parse_known_args()
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from server import make_handler
from tts_cpp import Engine
from request_schema import validate


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Engine(
            args.library,
            args.model,
            max_batch=4,
            quantization=args.quantization,
        )

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()

    def generate(self, payload, callback=None):
        is_batch = "requests" in payload
        requests, options = validate(payload, is_batch, self.engine.default_options())
        buffers = [bytearray() for _ in requests]

        def append(index, pcm):
            self.assertEqual(len(pcm), 3840)
            buffers[index].extend(pcm)

        stats = self.engine.generate(requests, options, callback or append)
        return [bytes(b) for b in buffers], stats

    def test_graph_eager_and_request_reset(self):
        payload = {
            "text": "你好，欢迎使用语音合成。",
            "max_frames": 4,
            "temperature": 0,
        }
        eager, _ = self.generate(dict(payload, use_cuda_graph=0))
        graph, stats = self.generate(payload)
        again, _ = self.generate(payload)
        self.assertEqual(eager, graph)
        self.assertEqual(graph, again)
        self.assertEqual(len(graph[0]), 4 * 3840)
        self.assertEqual(stats["truncated"], 1)

    def test_pool_stats_and_warm_reuse(self):
        payload = {"text": "你好。", "max_frames": 3, "temperature": 0}
        self.generate(payload)
        before = self.engine.memory_stats()
        self.generate(payload)
        after = self.engine.memory_stats()
        self.assertGreater(after["used_bytes"], 0)
        self.assertGreaterEqual(after["reserved_bytes"], after["used_bytes"])
        self.assertGreaterEqual(after["peak_used_bytes"], after["used_bytes"])
        self.assertGreaterEqual(after["peak_reserved_bytes"], after["reserved_bytes"])
        self.assertEqual(before["used_bytes"], after["used_bytes"])

        def callback(*_):
            with self.assertRaisesRegex(RuntimeError, "cannot reenter"):
                self.engine.memory_stats()
            return False

        self.generate(payload, callback)

    def test_seed_and_cfg(self):
        payload = {
            "text": "Hello, welcome aboard.",
            "instruction": "A calm, clear voice.",
            "cfg_scale": 4,
            "max_frames": 3,
            "seed": 91,
        }
        a, _ = self.generate(payload)
        b, _ = self.generate(payload)
        self.assertEqual(a, b)
        self.assertTrue(a[0])

    def test_graph_cache_across_prompt_lengths_and_shapes(self):
        # Captured pointers must survive arena reuse, prompt changes and key eviction.
        texts = [
            "你好。",
            "Welcome. This is a longer prompt for the same graph buffers.",
        ]
        for batch, cfg in [(1, 1), (2, 1), (2, 4), (1, 1)]:
            for text in texts:
                payload = {
                    "requests": [
                        {"text": text, "instruction": "A calm voice."}
                        for _ in range(batch)
                    ],
                    "max_frames": 3,
                    "temperature": 0,
                    "cfg_scale": cfg,
                }
                graph, _ = self.generate(payload)
                eager, _ = self.generate(dict(payload, use_cuda_graph=0))
                self.assertEqual(graph, eager)
        for k in range(1, 10):
            self.generate({"text": texts[0], "max_frames": 2, "top_k": k})
        payload = {"text": texts[1], "max_frames": 3, "temperature": 0}
        a, _ = self.generate(payload)
        b, _ = self.generate(dict(payload, use_cuda_graph=0))
        self.assertEqual(a, b)

    def test_batch_and_cancellation(self):
        payload = {
            "requests": [{"text": "你好。"}, {"text": "Welcome aboard."}],
            "max_frames": 4,
            "temperature": 0,
        }
        audio, _ = self.generate(payload)
        self.assertEqual(len(audio), 2)
        self.assertTrue(all(len(a) == 4 * 3840 for a in audio))
        self.assertNotEqual(audio[0], audio[1])

        def cancel(*_):
            raise InterruptedError("client disconnected")

        with self.assertRaises(InterruptedError):
            self.generate(payload, cancel)
        after, _ = self.generate(payload)
        self.assertEqual(audio, after)

    def test_batch_independent_eos(self):
        audio, stats = self.generate(
            {
                "requests": [{"text": "Hello."}, {"text": "Welcome aboard. " * 20}],
                "max_frames": 96,
                "temperature": 0.9,
                "seed": 42,
            }
        )
        self.assertGreater(len(audio[0]), 0)
        self.assertLess(len(audio[0]), len(audio[1]))
        # Quantization can change sampled EOS timing. Verify independent lane
        # completion and the frame cap without requiring BF16's sampled length.
        self.assertLessEqual(len(audio[1]), 96 * 3840)
        self.assertEqual(stats["frames"], len(audio[1]) // 3840)
        self.assertEqual(stats["truncated"], int(len(audio[1]) == 96 * 3840))

    def test_reject_invalid_and_context_overflow(self):
        for bad in [
            {"text": ""},
            {"text": "x", "cfg_scale": 4},
            {"text": "x", "max_frames": 2**40},
            {"text": "x", "temperature": float("nan")},
            {"text": "x\0y"},
        ]:
            with self.assertRaises(ValueError):
                validate(bad, False, self.engine.default_options())
        with self.assertRaises(RuntimeError):
            self.generate({"text": "测试" * 2048, "max_frames": 2000})

    def test_http_offline_stream_batch_busy(self):
        http_server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.engine))
        worker = threading.Thread(target=http_server.serve_forever)
        worker.start()

        def request(path, payload=None):
            conn = http.client.HTTPConnection(
                "127.0.0.1", http_server.server_port, timeout=60
            )
            conn.request(
                "GET" if payload is None else "POST",
                path,
                body=json.dumps(payload).encode() if payload is not None else None,
                headers={"Content-Type": "application/json"},
            )
            response = conn.getresponse()
            data = response.read()
            status = response.status
            conn.close()
            return status, data

        try:
            self.assertEqual(request("/health")[0], 200)
            p = {"text": "Hello.", "max_frames": 2, "temperature": 0}
            status, raw = request("/v1/audio/speech", p)
            self.assertEqual(status, 200)
            self.assertEqual(len(raw), 2 * 3840)
            status, data = request("/v1/audio/speech", dict(p, stream=False))
            self.assertEqual(status, 200)
            with wave.open(io.BytesIO(data)) as w:
                self.assertEqual(w.getframerate(), 24000)
                self.assertEqual(w.readframes(w.getnframes()), raw)
            status, data = request(
                "/v1/audio/batch",
                {"requests": [{"text": "你好。"}, {"text": "Hello."}], "max_frames": 2},
            )
            self.assertEqual(status, 200)
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                self.assertEqual(set(z.namelist()), {"0.wav", "1.wav", "metrics.json"})
            http_server.RequestHandlerClass.admission_gate.acquire()
            try:
                self.assertEqual(request("/v1/audio/speech", p)[0], 503)
            finally:
                http_server.RequestHandlerClass.admission_gate.release()
            self.assertEqual(request("/v1/audio/speech", {"text": ""})[0], 400)
        finally:
            http_server.shutdown()
            http_server.server_close()
            worker.join()


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]] + remaining, verbosity=2)
