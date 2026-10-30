"""HTTP response ownership only. Scheduling and bounded buffering live in C++."""

from collections import deque
import threading
import uuid


from request_schema import Overloaded


class Response:
    def __init__(self, owner, request_id, requests):
        self.owner, self.request_id, self.requests = owner, request_id, requests
        self.events = deque()
        self.done = self.cancelled = False
        self.stats = {}

    def cancel(self):
        with self.owner.condition:
            cancel = not self.done and not self.cancelled
            self.cancelled = True
            self.owner.jobs.discard(self)
        if cancel:
            self.owner.engine.cancel_request(self.request_id)

    def take(self, timeout=0.25):
        if self.events:
            return self.events.popleft()
        if self.done or self.cancelled:
            raise StopIteration
        try:
            frame = self.owner.engine.read_result(self.request_id, timeout)
        except Exception:
            # Native terminal errors consume the ID; do not cancel a later reuse.
            with self.owner.condition:
                self.done = True
                self.owner.jobs.discard(self)
            raise
        if not frame["ready"]:
            raise TimeoutError
        self.stats = {k: v for k, v in frame.items() if k != "chunks"}
        self.events.extend(frame["chunks"])
        self.done = frame["done"]
        if self.done:
            with self.owner.condition:
                self.owner.jobs.discard(self)
        if self.events:
            return self.events.popleft()
        if self.done:
            raise StopIteration
        raise TimeoutError


class Responses:
    def __init__(self, engine, *, queue_limit=32, buffer_bytes=1 << 20):
        engine.configure_serving(queue_limit, buffer_bytes)
        self.engine = engine
        self.condition = threading.Lock()
        self.jobs = set()
        self.stopping = False

    def submit(self, requests, options, prepare=None, request_id=None):
        requests = [dict(r) for r in requests]
        if prepare:
            prepare(requests)
        request_id = uuid.uuid4().hex if request_id is None else request_id
        with self.condition:
            if self.stopping:
                raise Overloaded("server is shutting down")
        response = Response(self, request_id, requests)
        self.engine.add_request(
            dict(request_id=request_id, inputs=requests, options=options)
        )
        with self.condition:
            if not self.stopping:
                self.jobs.add(response)
                return response
        # Close may finish while native submission is in flight. Cancel the
        # accepted ID without holding the HTTP bookkeeping lock.
        self.engine.cancel_request(request_id)
        raise Overloaded("server is shutting down")

    def close(self):
        with self.condition:
            self.stopping = True
            jobs = list(self.jobs)
        for job in jobs:
            job.cancel()
