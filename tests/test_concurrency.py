"""Native continuous batching and concurrent HTTP serving against the real checkpoint."""

from contextlib import contextmanager
import uuid
import argparse
import array
import os
import tempfile
import http.client
import json
import logging
from pathlib import Path
import sys
import threading
import unittest

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--library", type=Path)
parser.add_argument("--quantization", choices=("fp8", "int8"))
args, rest = parser.parse_known_args()
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "examples")]
from tts_cpp import Engine, configure_logging
from server import Server
from serving import Responses, Overloaded


@contextmanager
def submitted(engine, inputs, options, request_id=None):
    request_id = request_id or uuid.uuid4().hex
    assert (
        engine.add_request(dict(request_id=request_id, inputs=inputs, options=options))
        is None
    )
    try:
        yield request_id
    finally:
        engine.cancel_request(request_id)


class ConcurrencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Engine(
            args.library,
            args.model,
            max_batch=4,
            context=512,
            quantization=args.quantization,
        )
        cls.engine.run_async()

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()

    def test_multi_input_requests_share_admission_budget(self):
        # Five potential inputs compete for four execution rows. Each uses all
        # three guidance branches and the full page count. Their simultaneous
        # bound is four inputs, including when a later wave replaces a live row.
        with self.engine.reference_from_codes(
            [0] * self.engine.audio_format["codebooks"], "Hello."
        ) as reference:
            inputs = [dict(text="Hello.", instruction="Calm.", reference=reference)]
            options = dict(
                max_frames=480, dual_cfg=True, temperature=0, depth_temperature=0
            )
            with submitted(self.engine, inputs, options, "kv-occupant"):
                self.assertTrue(self.engine.read_result("kv-occupant")["chunks"])
                with submitted(self.engine, inputs * 4, options, "kv-waves"):
                    frame = self.engine.read_result("kv-waves")
                    self.assertTrue(frame["chunks"])
                    self.engine.cancel_request("kv-occupant")
                    emitted = {index for index, _ in frame["chunks"]}
                    while len(emitted) < 4 and not frame["done"]:
                        frame = self.engine.read_result("kv-waves")
                        emitted.update(index for index, _ in frame["chunks"])
                    self.assertEqual(emitted, set(range(4)))

    def test_padding_keeps_live_request_state(self):
        # Four requests have eleven CFG lanes. Replay pads one inactive lane.
        with self.engine.reference_from_codes(
            [0] * self.engine.audio_format["codebooks"],
            "Hello.",
        ) as reference:

            def run(graph):
                identities = [f"padding-{graph}-{i}" for i in range(4)]
                for i, identity in enumerate(identities):
                    options = dict(
                        max_frames=16,
                        temperature=0,
                        depth_temperature=0,
                        use_cuda_graph=graph,
                    )
                    options.update(dict(dual_cfg=True) if i < 3 else dict(cfg_scale=2))
                    self.engine.add_request(
                        dict(
                            request_id=identity,
                            inputs=[
                                dict(
                                    text="Welcome aboard.",
                                    instruction="Calm.",
                                    reference=reference,
                                )
                            ],
                            options=options,
                        )
                    )
                audio = []
                for identity in identities:
                    pcm = bytearray()
                    while True:
                        frame = self.engine.read_result(identity)
                        for _, chunk in frame["chunks"]:
                            pcm.extend(chunk)
                        if frame["done"]:
                            break
                    audio.append(bytes(pcm))
                return audio

            # Arrival timing can change GEMM geometry and near-tie argmax. The
            # native test compares exact PCM with a fixed four-request roster.
            for graph in (True, False):
                audio = run(graph)
                self.assertEqual(len(audio), 4)
                frame_bytes = self.engine.audio_format["samples_per_frame"] * 2
                for pcm in audio:
                    self.assertTrue(pcm)
                    self.assertEqual(len(pcm) % frame_bytes, 0)
                    self.assertLessEqual(len(pcm) // frame_bytes, 16)

    def test_arrival_preserves_inputs_and_initial_codes(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        os.environ["TTS_TOKENS_OUT"] = directory.name + "/codes-{request}.i32"
        self.addCleanup(os.environ.pop, "TTS_TOKENS_OUT", None)

        def tokens():
            paths = sorted(
                Path(directory.name).glob("*.i32"),
                key=lambda p: int(p.stem.split("-")[-1]),
            )
            return [p.read_bytes() for p in paths]

        with self.engine.reference_from_codes(
            [0] * self.engine.audio_format["codebooks"] * 2, "Hello."
        ) as reference:
            requests = [
                [dict(text="你好，这是第一段测试。")],
                [
                    dict(
                        text="Welcome aboard.",
                        instruction="A calm voice.",
                        reference=reference,
                    ),
                    dict(text="你好。", instruction="A calm voice."),
                ],
            ]
            options = [
                dict(max_frames=6, temperature=0, depth_temperature=0, seed=31),
                dict(
                    max_frames=4,
                    temperature=0,
                    depth_temperature=0,
                    seed=72,
                    cfg_scale=2,
                ),
            ]
            expected = [
                self.engine.generate(r, o)[0] for r, o in zip(requests, options)
            ]
            expected_tokens = tokens()
            sessions = ["arrival-first", "arrival-second"]
            self.engine.add_request(
                dict(request_id=sessions[0], inputs=requests[0], options=options[0])
            )
            actual = [[bytearray() for _ in r] for r in requests]
            first = self.engine.read_result(sessions[0])
            for lane, pcm in first["chunks"]:
                actual[0][lane].extend(pcm)
            self.engine.add_request(
                dict(request_id=sessions[1], inputs=requests[1], options=options[1])
            )
            with submitted(
                self.engine, [dict(text="Bad limit.")], dict(max_frames=999999)
            ) as invalid:
                with self.assertRaises(RuntimeError):
                    self.engine.read_result(invalid)
            try:
                live = set(range(2))
                while live:
                    for i in list(live):
                        frame = self.engine.read_result(sessions[i])
                        for lane, pcm in frame["chunks"]:
                            actual[i][lane].extend(pcm)
                        if frame["done"]:
                            live.remove(i)
                actual_tokens = tokens()[len(expected_tokens) :]
                books = self.engine.audio_format["codebooks"]
                for actual_codes, baseline_codes in zip(actual_tokens, expected_tokens):
                    a = array.array("i", actual_codes)
                    b = array.array("i", baseline_codes)
                    self.assertEqual(len(a), len(b))
                    self.assertEqual(a[:books], b[:books])
                    # Acoustic argmax can change at near ties when GEMM shape changes.
                    # Fixed-logit sampling and mapped KV/codec operators have exact tests.
                    self.assertTrue(
                        all(
                            0 <= code < self.engine.audio_format["codebook_size"]
                            for code in a
                        )
                    )
                # Codec GEMV/GEMM reduction precision also varies with active batch shape.
                for batch, reference in zip(actual, expected):
                    for pcm, baseline in zip(batch, reference):
                        a, b = array.array("h", pcm), array.array("h", baseline)
                        self.assertEqual(len(a), len(b))
                        # The first frame has identical codes; compare codec output alone.
                        n = self.engine.audio_format["samples_per_frame"]
                        # Use the same relative RMS tolerance as the HF codec oracle.
                        self.assertLess(
                            sum((x - y) ** 2 for x, y in zip(a[:n], b[:n])),
                            0.02**2 * max(1, sum(y * y for y in b[:n])),
                        )
            finally:
                for session in sessions:
                    self.engine.cancel_request(session)

    def test_cancel_releases_capacity_and_keeps_other_session_alive(self):
        options = dict(max_frames=64, temperature=0)
        first, second = "cancel-first", "cancel-second"
        self.engine.add_request(
            dict(
                request_id=first,
                inputs=[dict(text="Hello. " * 20)] * 3,
                options=options,
            )
        )
        self.engine.add_request(
            dict(
                request_id=second,
                inputs=[dict(text="你好。")],
                options=dict(max_frames=3, temperature=0),
            )
        )
        try:
            self.engine.read_result(first)
            self.engine.cancel_request(first)
            with submitted(
                self.engine, [dict(text="Reused capacity.")], dict(max_frames=2)
            ) as third:
                while not self.engine.read_result(third)["done"]:
                    pass
            self.assertTrue(self.engine.read_result(second)["chunks"])
        finally:
            self.engine.cancel_request(first)
            self.engine.cancel_request(second)

    def test_late_request_shares_native_batch(self):
        with submitted(
            self.engine, [dict(text="Welcome aboard. " * 20)], dict(max_frames=32)
        ) as long:
            self.assertTrue(self.engine.read_result(long)["chunks"])
            with submitted(
                self.engine, [dict(text="你好。")], dict(max_frames=3)
            ) as short:
                frames = []
                while True:
                    frame = self.engine.read_result(short)
                    frames.append(frame)
                    if frame["done"]:
                        break
                self.assertEqual(len(frames), 3)
                self.assertTrue(all(f["batch_size"] >= 2 for f in frames[1:]))
            # Native inference ran both while the caller only received the short response.
            self.assertGreaterEqual(self.engine.read_result(long)["frames"], 2)

    def test_engine_owns_requests_and_reuses_pool(self):
        engine = self.engine
        request, options = [dict(text="Hello.")], dict(max_frames=2, temperature=0)
        previous = None
        for _ in range(2):
            with submitted(engine, request, options, "reused") as request_id:
                while not engine.read_result(request_id)["done"]:
                    pass
            current = engine.memory_stats()
            if previous is not None:
                self.assertEqual(previous["used_bytes"], current["used_bytes"])
            previous = current

    def test_scheduler_and_gpu_executor_are_separate(self):
        threads = {}

        class Capture(logging.Handler):
            def emit(self, record):
                threads.setdefault(record.component, set()).add(threading.get_ident())

        logger = logging.Logger("worker-ownership")
        logger.addHandler(Capture())
        configure_logging("info", logger=logger, library=args.library)
        try:
            with submitted(
                self.engine,
                [dict(text="Welcome aboard. " * 20)],
                dict(max_frames=8, temperature=0),
            ) as request_id:
                self.assertTrue(self.engine.read_result(request_id)["chunks"])
                audio_format = self.engine.audio_format
                with self.engine.prepare_reference(
                    [0.0] * (audio_format["samples_per_frame"] * 2),
                    "Hello.",
                    sample_rate=audio_format["sample_rate"],
                ) as reference:
                    self.assertGreater(reference.frames, 0)
                    audio, _ = self.engine.generate(
                        [dict(text="你好。", reference=reference)],
                        dict(max_frames=2, temperature=0),
                    )
                    self.assertTrue(audio[0])
                while not self.engine.read_result(request_id)["done"]:
                    pass
            self.assertTrue(threads["reference"].isdisjoint(threads["request"]))
            self.assertEqual(len(threads["reference"]), 1)
            self.assertEqual(len(threads["request"]), 1)
            self.assertNotIn(threading.get_ident(), threads["request"])
            self.assertNotIn(threading.get_ident(), threads["reference"])
        finally:
            configure_logging("info", library=args.library)

    def test_z_shutdown_ends_outstanding_requests(self):
        # Last test: close the shared engine without loading a second copy of the model.
        self.engine.add_request(
            dict(
                request_id="closing",
                inputs=[dict(text="Hello.")],
                options=dict(max_frames=64),
            )
        )
        # Lifecycle belongs to one owner after API ingress has stopped.
        self.assertTrue(self.engine.read_result("closing")["chunks"])
        module, handle = self.engine._module, self.engine._handle
        self.engine.close()
        self.engine.close()
        with self.assertRaisesRegex(RuntimeError, "shut down"):
            module.read_result(handle, "closing", -1)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.engine.read_result("closing")

    def test_result_queries_timeout_while_executor_is_busy(self):
        entered, release = threading.Event(), threading.Event()
        errors = []

        class Capture(logging.Handler):
            def emit(self, record):
                if record.component == "reference":
                    entered.set()
                    release.wait(timeout=5)

        def encode():
            try:
                audio = self.engine.audio_format
                with self.engine.prepare_reference(
                    [0.0] * (audio["samples_per_frame"] * 2), "Hello.",
                    sample_rate=audio["sample_rate"],
                ):
                    pass
            except BaseException as error:
                errors.append(error)

        logger = logging.Logger("busy-executor")
        logger.addHandler(Capture())
        configure_logging("info", logger=logger, library=args.library)
        worker = threading.Thread(target=encode)
        worker.start()
        try:
            self.assertTrue(entered.wait(timeout=5))
            with submitted(self.engine, [dict(text="Hello.")], dict(max_frames=1), "query-timeout"):
                for timeout in (0, 0.01):
                    frame = self.engine.read_result("query-timeout", timeout)
                    self.assertFalse(frame["ready"])
                    self.assertEqual(frame["request_id"], "query-timeout")
                release.set()
                self.assertTrue(self.engine.read_result("query-timeout")["done"])
        finally:
            release.set()
            worker.join(timeout=10)
            configure_logging("warn", library=args.library)
        self.assertFalse(worker.is_alive())
        self.assertFalse(errors, errors)

    def test_identity_terminal_cleanup_and_errors(self):
        engine = self.engine
        request = dict(
            request_id="identity",
            inputs=[dict(text="Hello.")],
            options=dict(max_frames=2, temperature=0),
        )
        self.assertIsNone(engine.add_request(request))
        try:
            self.assertIsNone(engine.add_request(request))
            with self.assertRaisesRegex((RuntimeError, ValueError), "duplicate"):
                engine.read_result("identity")
            request["inputs"][0]["text"] = ""
            while True:
                frame = engine.read_result("identity", 0.1)
                self.assertEqual(frame["request_id"], "identity")
                if frame["done"]:
                    break
            with self.assertRaisesRegex(
                (RuntimeError, ValueError), "unknown request_id"
            ):
                engine.read_result("identity", 0)
            self.assertIsNone(engine.cancel_request("identity"))
            request["inputs"][0]["text"] = "Hello."
            engine.add_request(request)
            self.assertIsNone(engine.cancel_request("identity"))
            self.assertIsNone(engine.cancel_request("identity"))
            with self.assertRaisesRegex((RuntimeError, ValueError), "unknown request_id"):
                engine.read_result("identity", 0)
            request["options"]["max_frames"] = 999999
            engine.add_request(request)
            with self.assertRaises(RuntimeError):
                engine.read_result("identity")
            self.assertIsNone(engine.cancel_request("identity"))
        finally:
            engine.cancel_request("identity")

    def test_completed_unread_output_counts_toward_admission(self):
        engine = self.engine
        engine.configure_serving(2)
        options = dict(max_frames=1, temperature=0)
        try:
            for request_id in ("unread", "barrier"):
                engine.add_request(
                    dict(
                        request_id=request_id,
                        inputs=[dict(text="Hello.")],
                        options=options,
                    )
                )
            self.assertTrue(engine.read_result("barrier")["done"])
            engine.configure_serving(1)
            request = dict(
                request_id="replacement", inputs=[dict(text="Hello.")], options=options
            )
            with self.assertRaisesRegex(RuntimeError, "queue is full"):
                engine.add_request(request)
            self.assertTrue(engine.read_result("unread")["done"])
            engine.add_request(request)
            self.assertTrue(engine.read_result("replacement")["done"])
        finally:
            for request_id in ("unread", "barrier", "replacement"):
                engine.cancel_request(request_id)
            engine.configure_serving()

    def test_control_and_submission_do_not_wait_for_scheduler(self):
        entered, release, returned = (threading.Event() for _ in range(3))
        errors = []

        class Capture(logging.Handler):
            def emit(self, record):
                if record.component == "request" and not entered.is_set():
                    entered.set()
                    release.wait(timeout=5)

        logger = logging.Logger("submission-with-busy-scheduler")
        logger.addHandler(Capture())
        configure_logging("info", logger=logger, library=args.library)

        def submit():
            try:
                self.engine.cancel_request("unknown-control-target")
                self.engine.add_request(dict(request_id="posted", inputs=[dict(text="Hello.")],
                                             options=dict(max_frames=1, temperature=0)))
            except BaseException as error:
                errors.append(error)
            finally:
                returned.set()

        worker = threading.Thread(target=submit)
        try:
            self.engine.add_request(dict(request_id="scheduler-held", inputs=[dict(text="Hello.")],
                                         options=dict(max_frames=1, temperature=0)))
            self.assertTrue(entered.wait(timeout=5))
            worker.start()
            self.assertTrue(returned.wait(timeout=1), "control or submission waited for the scheduler")
        finally:
            release.set()
            if worker.ident is not None:
                worker.join(timeout=5)
            configure_logging("warn", library=args.library)
        self.assertFalse(errors, errors)
        for request_id in ("scheduler-held", "posted"):
            self.assertTrue(self.engine.read_result(request_id)["done"])

    def test_failed_response_does_not_cancel_reused_id(self):
        responses = Responses(self.engine)
        try:
            failed = responses.submit(
                [dict(text="Hello.")], dict(max_frames=999999), request_id="reused-http"
            )
            with self.assertRaises(RuntimeError):
                self.drain(failed)
            good = responses.submit(
                [dict(text="Hello.")], dict(max_frames=2), request_id="reused-http"
            )
            failed.cancel()
            self.assertEqual(len(self.drain(good)), 2)
        finally:
            responses.close()

    @staticmethod
    def drain(job):
        chunks = []
        while True:
            try:
                chunks.append(job.take())
            except TimeoutError:
                continue
            except StopIteration:
                return chunks

    def test_bounded_slow_consumer_does_not_block_other_request(self):
        scheduler = Responses(
            self.engine,
            queue_limit=2,
            buffer_bytes=self.engine.audio_format["samples_per_frame"] * 4,
        )
        try:
            options = self.engine.default_options()
            options.max_frames = 12
            options.temperature = options.depth_temperature = 0
            slow = scheduler.submit([dict(text="Welcome aboard. " * 20)], options)
            options.max_frames = 2
            fast = scheduler.submit([dict(text="你好。")], options)
            with self.assertRaises(Overloaded):
                scheduler.submit([dict(text="Full.")], options)
            self.assertEqual(len(self.drain(fast)), 2)
            # Leave this response unread until native bounded buffering cancels it.
            import time

            time.sleep(0.5)
            with self.assertRaisesRegex(RuntimeError, "buffer limit"):
                self.drain(slow)
        finally:
            scheduler.close()
            self.engine.configure_serving()
        # Shutdown and cancellation return the native session capacity.
        with submitted(
            self.engine, [dict(text="Hello.")] * 4, dict(max_frames=1)
        ) as session:
            emitted = set()
            while True:
                frame = self.engine.read_result(session)
                emitted.update(index for index, _ in frame["chunks"])
                if frame["done"]:
                    break
            self.assertEqual(emitted, set(range(4)))

    def test_late_http_request_finishes_before_long_request(self):
        server = Server(("127.0.0.1", 0), self.engine)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        long = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=60)
        short = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=60)
        try:
            text = "Welcome aboard. This is a long streaming request. " * 15
            long.request(
                "POST",
                "/v1/audio/speech",
                json.dumps(dict(text=text, max_frames=64, temperature=0)),
                {"Content-Type": "application/json"},
            )
            response = long.getresponse()
            self.assertEqual(response.status, 200)
            self.assertTrue(
                response.read(self.engine.audio_format["samples_per_frame"] * 2)
            )
            with server.responses.condition:
                job = next(
                    j for j in server.responses.jobs if j.requests[0]["text"] == text
                )
            short.request(
                "POST",
                "/v1/audio/speech",
                json.dumps(
                    dict(text="你好。", max_frames=2, temperature=0, stream=False)
                ),
                {"Content-Type": "application/json"},
            )
            result = short.getresponse()
            self.assertEqual(result.status, 200)
            self.assertTrue(result.read().startswith(b"RIFF"))
            self.assertFalse(
                job.done, "the late request waited for the whole long request"
            )
        finally:
            long.close()
            short.close()
            server.shutdown()
            server.server_close()
            thread.join()

    def test_http_queue_limit_and_shutdown(self):
        server = Server(("127.0.0.1", 0), self.engine, queue_limit=1)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        first = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=60)
        second = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=60)
        try:
            payload = dict(text="Welcome aboard. " * 20, max_frames=64, temperature=0)
            first.request(
                "POST",
                "/v1/audio/speech",
                json.dumps(payload),
                {"Content-Type": "application/json"},
            )
            self.assertEqual(first.getresponse().status, 200)
            second.request(
                "POST",
                "/v1/audio/speech",
                json.dumps(dict(text="Hello.")),
                {"Content-Type": "application/json"},
            )
            response = second.getresponse()
            self.assertEqual(response.status, 503)
            self.assertIn("queue", json.loads(response.read())["error"])
        finally:
            first.close()
            second.close()
            server.shutdown()
            server.server_close()
            thread.join()
        self.assertTrue(server.responses.stopping)
        self.engine.configure_serving()


class LifecycleTests(unittest.TestCase):
    def test_total_memory_limit_rejects_loading(self):
        with self.assertRaisesRegex(RuntimeError, "memory budget exceeded"):
            Engine(
                args.library,
                args.model,
                max_batch=1,
                context=256,
                quantization=args.quantization,
                memory_limit_mb=8,
            )

    def test_fixed_pool_admission_failure_and_recovery(self):
        with Engine(
            args.library,
            args.model,
            max_batch=1,
            context=256,
            quantization=args.quantization,
            runtime_memory_mb=32,
        ) as engine:
            engine.run_async()
            initial = engine.memory_stats()
            inputs = [dict(text="Hello.")]
            options = dict(max_frames=4, temperature=0, depth_temperature=0)
            original, _ = engine.generate(inputs, options)
            delivered = []
            with self.assertRaisesRegex(RuntimeError, "insufficient GPU memory"):
                engine.generate(
                    [dict(text="Hello."), dict(text="Hello. " * 85)],
                    options,
                    lambda index, pcm: delivered.append(index),
                )
            self.assertFalse(
                delivered, "a later input was rejected after partial audio"
            )
            waves, _ = engine.generate(inputs * 3, dict(options, max_frames=1))
            self.assertEqual(len(waves), 3)
            self.assertTrue(
                all(waves), "reserved KV credits did not survive input waves"
            )
            with engine.reference_from_codes(
                [0] * engine.audio_format["codebooks"],
                "Hello.",
            ) as reference:
                with self.assertRaisesRegex(RuntimeError, "insufficient GPU memory"):
                    engine.generate(
                        [dict(text="Hello.", instruction="Calm.", reference=reference)],
                        dict(options, dual_cfg=True),
                    )
            after, _ = engine.generate(inputs, options)
            self.assertEqual(after, original)
            self.assertEqual(engine.memory_stats(), initial)

    def test_explicit_start_and_terminal_close(self):
        # Entering a context manages ownership; it must not start execution.
        with Engine(
            args.library,
            args.model,
            max_batch=1,
            context=256,
            quantization=args.quantization,
        ) as engine:
            self.assertGreater(engine.audio_format["sample_rate"], 0)
            request = dict(
                request_id="lifecycle",
                inputs=[dict(text="Hello.")],
                options=dict(max_frames=2, temperature=0),
            )
            with self.assertRaisesRegex(RuntimeError, "run_async"):
                engine.add_request(request)
            self.assertIsNone(engine.run_async())
            self.assertIsNone(engine.run_async())
            audio, _ = engine.generate(request["inputs"], request["options"])
            self.assertTrue(audio[0])
        engine.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            engine.run_async()

    def test_close_without_start(self):
        with Engine(
            args.library,
            args.model,
            max_batch=1,
            context=256,
            quantization=args.quantization,
        ) as engine:
            engine.close()
            engine.close()
        # Dropping an initialized but unstarted native engine must also be safe.
        engine = Engine(
            args.library,
            args.model,
            max_batch=1,
            context=256,
            quantization=args.quantization,
        )
        del engine


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], *rest], verbosity=2)
