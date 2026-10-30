"""HTTP response ownership under concurrent native calls; no GPU required."""

from pathlib import Path
import sys
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from serving import Overloaded, Responses


class StubEngine:
    def __init__(self, submit):
        self.add_request = submit
        self.cancelled = []

    def configure_serving(self, *_):
        pass

    def cancel_request(self, request_id):
        self.cancelled.append(request_id)


class ResponseOwnershipTests(unittest.TestCase):
    def test_submissions_can_enter_native_engine_concurrently(self):
        entered = threading.Barrier(2)
        engine = StubEngine(lambda _: entered.wait(timeout=2))
        responses = Responses(engine)
        errors = []

        def submit(identity):
            try:
                responses.submit([dict(text="Hello.")], {}, request_id=identity)
            except BaseException as error:
                errors.append(error)

        workers = [threading.Thread(target=submit, args=(str(i),)) for i in range(2)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=5)
            self.assertFalse(worker.is_alive())
        self.assertFalse(errors, errors)
        responses.close()
        self.assertCountEqual(engine.cancelled, ["0", "1"])

    def test_close_cancels_a_submission_that_finishes_after_close(self):
        entered, release = threading.Event(), threading.Event()
        errors = []

        def native_submit(_):
            entered.set()
            if not release.wait(timeout=5):
                raise TimeoutError("submission was not released")

        engine = StubEngine(native_submit)
        responses = Responses(engine)

        def submit():
            try:
                responses.submit([dict(text="Hello.")], {}, request_id="closing")
            except BaseException as error:
                errors.append(error)

        worker = threading.Thread(target=submit)
        closer = threading.Thread(target=responses.close)
        worker.start()
        try:
            self.assertTrue(entered.wait(timeout=2))
            closer.start()
            closer.join(timeout=2)
            self.assertFalse(closer.is_alive(), "close waited for a native submission")
        finally:
            release.set()
            worker.join(timeout=5)
            if closer.ident is not None:
                closer.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], Overloaded)
        self.assertEqual(engine.cancelled, ["closing"])
        self.assertFalse(responses.jobs)


if __name__ == "__main__":
    unittest.main()
