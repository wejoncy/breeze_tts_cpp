"""Native/Python logging integration, including callback reentry and lifetime."""

import argparse
import gc
import json
import logging
from pathlib import Path
import sys
import tempfile
import unittest

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=Path, required=True)
args, remaining = parser.parse_known_args()
sys.argv = [sys.argv[0], *remaining]

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from tts_cpp import Engine, configure_logging


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []
        self.engine = None
        self.failures = []

    def emit(self, record):
        self.records.append(record)
        calls = [lambda: configure_logging("off")]
        if self.engine is not None:
            calls.extend([
                self.engine.memory_stats,
                self.engine.run_async,
                self.engine.close,
                self.engine.configure_serving,
                lambda: self.engine.add_request({}),
                lambda: self.engine.read_result("callback", 0),
                lambda: self.engine.cancel_request("callback"),
            ])
        for call in calls:
            try:
                call()
            except RuntimeError as error:
                if "cannot reenter" not in str(error):
                    self.failures.append(str(error))
            else:
                self.failures.append("callback reentry was accepted")


class LoggingTests(unittest.TestCase):
    def tearDown(self):
        configure_logging("info")

    def test_python_callback_and_engine_context(self):
        logger = logging.Logger("native-test", level=5)
        capture = Capture()
        logger.addHandler(capture)
        configure_logging("debug", logger=logger)
        gc.collect()
        with Engine(
            model=args.model, max_batch=1, quantization="int8"
        ) as engine:
            engine.run_async()
            capture.engine = engine
            # The caller waits for the worker while holding the Python engine lock.
            # Log reentry must reject before trying that lock on the worker thread.
            with engine.prepare_reference(
                [0.0] * (engine.audio_format["samples_per_frame"] * 2),
                "Hello.", sample_rate=engine.audio_format["sample_rate"],
            ):
                pass
            a, _ = engine.generate(
                [dict(text="你好。")], dict(max_frames=2, temperature=0)
            )
            first_id = [
                r.request_id
                for r in capture.records
                if r.component == "request" and "complete" in r.msg
            ][-1]
            b, _ = engine.generate(
                [dict(text="你好。")], dict(max_frames=2, temperature=0)
            )
            self.assertEqual(a, b)
            completed = [
                r
                for r in capture.records
                if r.component == "request" and "complete" in r.msg
            ]
            self.assertGreater(completed[-1].request_id, first_id)
            self.assertEqual(len({r.engine_id for r in capture.records}), 1)
            self.assertGreater(capture.records[0].engine_id, 0)
            self.assertTrue(all(r.timestamp_us > 0 for r in capture.records))
            self.assertFalse(capture.failures)
            count = len(capture.records)
            configure_logging("off")
            engine.generate([dict(text="你好。")], dict(max_frames=2, temperature=0))
            self.assertEqual(len(capture.records), count)
            configure_logging("info", logger=logger)
            engine.generate([dict(text="你好。")], dict(max_frames=2, temperature=0))
            self.assertGreater(len(capture.records), count)
            capture.engine = None

    def test_json_errors_and_failed_reconfiguration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "native.jsonl"
            configure_logging("error", file=path, json=True)
            with self.assertRaises(ValueError):
                configure_logging("garbage")
            with self.assertRaises(RuntimeError):
                configure_logging("info", file=directory)
            with self.assertRaises(RuntimeError):
                Engine(model=directory)
            configure_logging("off")
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertTrue(records)
            self.assertTrue(all(r["level"] == "error" for r in records))
            self.assertTrue(all(r["component"] == "api" for r in records))

    def test_c_callback_failure_cannot_fail_inference(self):
        class Broken(logging.Handler):
            def emit(self, _):
                raise RuntimeError("broken destination")

        logger = logging.Logger("broken")
        logger.addHandler(Broken())
        configure_logging("info", logger=logger)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                RuntimeError, "config|model|file|open|directory"
            ):
                Engine(model=directory)


if __name__ == "__main__":
    unittest.main()
