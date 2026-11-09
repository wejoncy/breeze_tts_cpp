"""Own original HTTP response connections and drain the shared Engine output FIFO."""

from collections import deque
import threading
import time
import uuid

from request_schema import Overloaded


class Response:
    def __init__(self, owner, request_id, requests):
        self.owner, self.request_id, self.requests = owner, request_id, requests
        self.frames, self.events = deque(), deque()
        self.buffered_bytes = 0
        self.done = self.cancelled = self.output_failed = False
        self.stats = {}

    def cancel(self):
        with self.owner.condition:
            cancel = not self.done and not self.cancelled
            self.cancelled = True
            self.owner.jobs.pop(self.request_id, None)
            self.frames.clear()
            self.buffered_bytes = 0
            self.owner.condition.notify_all()
        if cancel:
            self.owner.engine.cancel_request(self.request_id)

    def take(self, timeout=0.25):
        if self.events:
            return self.events.popleft()
        if self.done or self.cancelled:
            raise StopIteration
        try:
            frame = self.owner.take(self, timeout)
        except TimeoutError:
            raise
        except Exception:
            self.cancel()
            raise
        self.done = frame["done"]
        if self.done:
            with self.owner.condition:
                self.owner.jobs.pop(self.request_id, None)
        if frame.get("error"):
            raise RuntimeError(frame["error"])
        if frame.get("cancelled"):
            self.cancelled = True
            raise RuntimeError("generation cancelled")
        self.stats = {k: v for k, v in frame.items() if k != "chunks"}
        self.events.extend(frame["chunks"])
        if self.events:
            return self.events.popleft()
        if self.done:
            raise StopIteration
        raise TimeoutError


class Responses:
    def __init__(self, engine, *, queue_limit=32, buffer_bytes=1 << 20):
        engine.configure_serving(queue_limit, buffer_bytes * queue_limit)
        self.engine = engine
        self.condition = threading.Condition()
        self.drain = threading.Lock()
        self.buffer_bytes = buffer_bytes
        self.queue_limit = queue_limit
        self.jobs = {}
        self.stopping = False

    def take(self, response, timeout):
        deadline = time.monotonic() + timeout
        while True:
            with self.condition:
                if response.frames:
                    frame = response.frames.popleft()
                    response.buffered_bytes -= sum(len(pcm) for _, pcm in frame["chunks"])
                    return frame
                if response.cancelled or self.stopping:
                    raise RuntimeError("HTTP response is closed")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError
                if not self.drain.acquire(blocking=False):
                    self.condition.wait(remaining)
                    continue
            cancel = None
            try:
                frame = self.engine.output_queue.get(remaining)
                with self.condition:
                    target = self.jobs.get(frame["request_id"])
                    if target is not None and not target.output_failed:
                        size = sum(len(pcm) for _, pcm in frame["chunks"])
                        if size > self.buffer_bytes - target.buffered_bytes:
                            target.frames.clear()
                            target.buffered_bytes = 0
                            target.output_failed = True
                            cancel = target.request_id
                            frame = dict(request_id=cancel, chunks=[], done=True,
                                         error="HTTP audio consumer exceeded its buffer limit")
                        else:
                            target.buffered_bytes += size
                        target.frames.append(frame)
            finally:
                with self.condition:
                    self.drain.release()
                    self.condition.notify_all()
            if cancel is not None:
                self.engine.cancel_request(cancel)

    def submit(self, requests, options, prepare=None, request_id=None):
        requests = [dict(r) for r in requests]
        if prepare:
            prepare(requests)
        request_id = uuid.uuid4().hex if request_id is None else request_id
        response = Response(self, request_id, requests)
        # Register the original connection before publishing its request. Holding
        # this lock prevents dispatch or close racing the nonblocking enqueue.
        with self.condition:
            if self.stopping:
                raise Overloaded("server is shutting down")
            if len(self.jobs) >= self.queue_limit:
                raise Overloaded("HTTP request capacity reached")
            if request_id in self.jobs:
                raise ValueError("duplicate HTTP request ID")
            self.jobs[request_id] = response
            try:
                self.engine.add_request(dict(request_id=request_id, inputs=requests, options=options))
            except Exception:
                self.jobs.pop(request_id, None)
                raise
        return response

    def close(self):
        with self.condition:
            self.stopping = True
            jobs = list(self.jobs.values())
            self.condition.notify_all()
        for job in jobs:
            job.cancel()
