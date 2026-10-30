"""Checkpoint integration tests for native reference/batch/API parity."""

import argparse
import http.client
import io
import json
import math
import sys
import threading
import unittest
import wave
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--library", type=Path)
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--quantization", choices=("fp8", "int8"))
args, remaining = parser.parse_known_args()
sys.argv = [sys.argv[0], *remaining]
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "examples")]
from server import Server
from tts_cpp import Engine
from request_schema import validate


class ParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Engine(
            library=args.library,
            model=args.model,
            max_batch=4,
            quantization=args.quantization,
        )
        cls.engine.run_async()

        def fixture(name, text):
            path = ROOT / "artifacts/final" / name
            if path.exists():
                return path.read_bytes()
            pcm, _ = cls.engine.generate(
                [{"text": text}], {"max_frames": 24, "temperature": 0}
            )
            output = io.BytesIO()
            with wave.open(output, "wb") as f:
                f.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                f.writeframes(pcm[0])
            return output.getvalue()

        cls.audio = fixture(
            "chinese.wav",
            "你好，欢迎使用语音合成。这段声音由原生引擎生成，支持批量处理和实时流式输出。",
        )
        cls.ref = cls.engine.prepare_reference(
            cls.audio,
            "你好，欢迎使用语音合成。这段声音由原生引擎生成，支持批量处理和实时流式输出。",
        )
        cls.ref2 = cls.engine.prepare_reference(
            fixture("english.wav", "Hello, welcome aboard."), "Hello, welcome aboard."
        )

    @classmethod
    def tearDownClass(cls):
        cls.ref.close()
        cls.ref2.close()
        cls.engine.close()

    def run_pair(self, requests, **kwargs):
        opts = dict(max_frames=4, temperature=0, **kwargs)
        a, stats = self.engine.generate(requests, opts)
        b, _ = self.engine.generate(requests, dict(opts, use_cuda_graph=0))
        self.assertEqual(a, b)
        self.assertTrue(all(len(x) == 4 * 3840 for x in a))
        return a

    def test_reference_batch_mixed_and_reuse(self):
        requests = [
            dict(text="你好，今天很高兴见到你。", reference=self.ref),
            dict(text="Hello.", reference=self.ref2),
            dict(text="欢迎使用语音合成。"),
            dict(text="Welcome aboard.", reference=self.ref, speaker="S1"),
        ]
        a = self.run_pair(requests)
        self.assertEqual(len(set(a)), 4)
        b, _ = self.engine.generate(requests, dict(max_frames=4, temperature=0))
        self.assertEqual(a, b)
        imported = self.engine.reference_from_codes(
            self.ref.codes(),
            "你好，欢迎使用语音合成。这段声音由原生引擎生成，支持批量处理和实时流式输出。",
        )
        try:
            x, _ = self.engine.generate(
                [requests[0]], dict(max_frames=4, temperature=0)
            )
            y, _ = self.engine.generate(
                [dict(requests[0], reference=imported)],
                dict(max_frames=4, temperature=0),
            )
            self.assertEqual(x, y)
        finally:
            imported.close()

    def test_direction_and_dual_batch(self):
        requests = [
            dict(text=t, instruction="A calm, clear voice.", reference=r)
            for t, r in [("你好。", self.ref), ("Welcome aboard.", self.ref2)]
        ]
        self.run_pair(requests, cfg_scale=0.5)
        self.run_pair(requests, cfg_scale=3)
        self.run_pair(requests, cfg_scale_ref=2.0, cfg_scale_ins=3.0)
        self.run_pair(requests, cfg_scale_ref=0.0, cfg_scale_ins=1.0)

    def test_sampling_flags_cache_and_speakers(self):
        requests = [
            dict(text="[laughter] Hello.", speaker=""),
            dict(text="欢迎。[laughter]", speaker="[S3]"),
        ]
        for i in range(10):
            opts = dict(
                max_frames=4,
                temperature=0.5 + 0.03 * i,
                top_p=0.8,
                top_k=32,
                depth_top_p=0.7,
                depth_top_k=12,
                depth_temperature=0.7,
                repetition_penalty=1.0 + i * 0.03,
                seed=91,
            )
            a, _ = self.engine.generate(requests, opts)
            b, _ = self.engine.generate(requests, dict(opts, use_cuda_graph=0))
            self.assertEqual(a, b)
        a, _ = self.engine.generate(
            requests, dict(max_frames=4, do_sample=False, depth_do_sample=False, seed=1)
        )
        b, _ = self.engine.generate(
            requests, dict(max_frames=4, temperature=0, seed=999)
        )
        self.assertEqual(a, b)

    def test_cancel_reset_and_microbatch_128(self):
        requests = [
            dict(text="你好。", reference=self.ref),
            dict(text="Hello.", reference=self.ref2),
        ]
        opts = dict(max_frames=4, temperature=0)
        a, _ = self.engine.generate(requests, opts)
        st = self.engine.generate(requests, opts, lambda i, pcm: False)
        self.assertTrue(st["cancelled"])
        b, _ = self.engine.generate(requests, opts)
        self.assertEqual(a, b)
        pcm, stats = self.engine.generate(
            [dict(text="Hi.") for _ in range(128)], dict(max_frames=1, temperature=0)
        )
        self.assertEqual(len(pcm), 128)
        self.assertTrue(all(len(x) == 3840 for x in pcm))

    def test_nonaligned_pcm_and_resampling(self):
        # Exercise convolution ceil sizing, single-sample input, and native soxr.
        for count, rate, channels in [
            (1, 24000, 1),
            (19201, 24000, 1),
            (70013, 44100, 2),
        ]:
            pcm = [
                0.1 * math.sin(2 * math.pi * 220 * i / rate)
                for i in range(count)
                for _ in range(channels)
            ]
            with self.engine.prepare_reference(
                pcm, "Reference.", sample_rate=rate, channels=channels
            ) as ref:
                expected = math.ceil(math.ceil(count * 24000 / rate) / 1920)
                self.assertEqual(ref.frames, expected)
                self.assertTrue(all(0 <= c < 2048 for c in ref.codes()))

    def test_errors(self):
        for values in [
            dict(top_p=0),
            dict(depth_top_p=2),
            dict(repetition_penalty=0),
            dict(cfg_scale_ref=2),
            dict(cfg_scale_ins=float("nan")),
            dict(do_sample=2),
            dict(temperature=1e100),
        ]:
            with self.assertRaises(ValueError):
                validate(
                    dict(text="Hello.", **values), False, self.engine.default_options()
                )
        with self.assertRaises(RuntimeError):
            self.engine.prepare_reference(b"bad audio", "test")
        with self.assertRaises(RuntimeError):
            self.engine.prepare_reference(
                [float("nan")] * 1920, "test", sample_rate=24000
            )
        closed = self.engine.reference_from_codes(self.ref.codes(), "test")
        closed.close()
        with self.assertRaises(ValueError):
            self.engine.generate([dict(text="Hello.", reference=closed)])

    def test_http_multipart_batch_model(self):
        server = Server(("127.0.0.1", 0), self.engine)
        worker = threading.Thread(target=server.serve_forever)
        worker.start()

        def call(path, fields=None, audio=None):
            conn = http.client.HTTPConnection(
                "127.0.0.1", server.server_port, timeout=60
            )
            body = None
            if fields is not None:
                parts = []
                for key, value in fields.items():
                    parts.append(
                        f'--ttsboundary\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode()
                    )
                if audio:
                    parts.append(
                        b'--ttsboundary\r\nContent-Disposition: form-data; name="ref_audio"; filename="reference.wav"\r\nContent-Type: audio/wav\r\n\r\n'
                        + audio
                        + b"\r\n"
                    )
                body = b"".join(parts) + b"--ttsboundary--\r\n"
            conn.request(
                "POST" if fields is not None else "GET",
                path,
                body,
                {"Content-Type": "multipart/form-data; boundary=ttsboundary"},
            )
            r = conn.getresponse()
            data = r.read()
            result = (r.status, dict(r.getheaders()), data)
            conn.close()
            return result

        try:
            status, _, data = call("/v1/model")
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(data)["max_batch_texts"], 128)
            self.assertEqual(len(json.loads(data)["model_digest"]), 64)
            fields = dict(
                text="你好。",
                ref_text="你好，欢迎使用语音合成。",
                max_new_tokens=3,
                temperature=0,
            )
            status, h, data = call("/v1/audio/speech", fields, self.audio)
            self.assertEqual(status, 200, data)
            self.assertEqual(len(data), 3 * 3840)
            self.assertEqual(h["X-Sample-Rate"], "24000")
            fields = dict(
                texts=json.dumps(
                    ["你好。", "Welcome.", "Hello.", "再见。", "Batch five."]
                ),
                ref_text="你好，欢迎使用语音合成。",
                max_new_tokens=2,
                temperature=0,
            )
            status, h, data = call("/v1/audio/speech/batch", fields, self.audio)
            self.assertEqual(status, 200, data)
            self.assertEqual(h["X-Segment-Bytes"], ",".join(["7680"] * 5))
            self.assertEqual(len(data), 5 * 7680)
            self.assertEqual(
                call("/v1/audio/speech", dict(text="Hi.", ref_text="missing file"))[0],
                400,
            )
        finally:
            server.shutdown()
            server.server_close()
            worker.join()


if __name__ == "__main__":
    unittest.main(verbosity=2)
