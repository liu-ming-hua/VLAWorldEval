"""Batch independent native replay RPCs while reusing ordinary replay reports.

The model has no per-episode mutable state: each request explicitly carries its
own terminal image, action, seed and artifact directory. Only the broker calls
the subprocess client, which is deliberately not thread-safe.
"""
from concurrent.futures import Future
from dataclasses import asdict
import json
from pathlib import Path
from queue import Empty, Queue
from threading import Lock, Thread
import time

import numpy as np

from evaluation.execution.client import ModelWorker
from evaluation.execution.worker_session import NativeWorkerSession


class BatchedNativeSession(NativeWorkerSession):
    def __init__(self, directory, *, batch_size, on_generated=None):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Positive batch size required")
        super().__init__(directory, on_generated=on_generated)
        self.batch_size = batch_size
        self.lock = Lock()
        self.queue = Queue()
        self.broker = None
        self.batch_error = None
        self.dispatches = 0
        self.generated_chunks = 0

    def acquire(self, python, *, module, adapter, kwargs, gpus, memory_gib, options):
        identity = json.dumps({"python": str(python), "module": module, "adapter": adapter,
            "kwargs": {k: v for k, v in kwargs.items() if k != "runtime_dir"},
            "gpus": gpus, "memory_gib": memory_gib, "options": asdict(options)}, sort_keys=True)
        with self.lock:
            if self.closed or self.batch_error is not None:
                raise RuntimeError("Batch session is closed or failed")
            reused = self.worker is not None
            if reused and identity != self.identity:
                raise ValueError("Cannot mix checkpoint/protocol/runtime configurations in a batch")
            if not reused:
                if adapter != "Cosmos3DroidFDNativeAdapter" or kwargs.get("diffusion_cache") is not False:
                    raise ValueError("Only qualified stateless native Cosmos batching is enabled")
                self.identity = identity
                self.worker = ModelWorker(python, self.directory/"worker", module=module,
                    adapter=adapter, kwargs=kwargs, gpus=gpus, memory_gib=memory_gib, options=options)
                self.broker = Thread(target=self._serve_batches, daemon=True)
                self.broker.start()
            self.cases += 1
            return _EpisodeProxy(self, Path(kwargs["runtime_dir"])), reused

    def _serve_batches(self):
        while True:
            first = self.queue.get()
            if first is None:
                return
            pending = [first]
            deadline = time.monotonic()+0.05
            while len(pending) < self.batch_size:
                try:
                    item = self.queue.get(timeout=max(0, deadline-time.monotonic()))
                except Empty:
                    break
                if item is None:
                    raise RuntimeError("Session closed with requests still active")
                pending.append(item)
            if self.batch_error is not None:
                for _, future in pending:
                    future.set_exception(self.batch_error)
                continue
            try:
                started = time.perf_counter()
                results = self.worker.call("native_batch_step", items=[i for i, _ in pending])
                if len(results) != len(pending):
                    raise ValueError("Native batch output count mismatch")
                self.dispatches += 1
                self.generated_chunks += len(pending)
                with (self.directory/"dispatches.jsonl").open("a") as log:
                    log.write(json.dumps({"batch_size": len(pending), "seconds": time.perf_counter()-started,
                        "generated_chunks": self.generated_chunks, "memory": self.worker.memory})+"\n")
                for (_, future), result in zip(pending, results):
                    future.set_result(result)
            except BaseException as exc:
                self.batch_error = exc
                for _, future in pending:
                    future.set_exception(exc)

    def track_report(self, path):
        with self.lock:
            super().track_report(path)

    def close(self):
        if self.broker is not None and self.broker.is_alive():
            self.queue.put(None)
            self.broker.join()
        super().close()


class _EpisodeProxy:
    def __init__(self, session, runtime_dir):
        self.session = session
        self.runtime_dir = runtime_dir
        self.ready = dict(session.worker.ready)
        self.image = None
        self.seed = None
        self.index = 0

    @property
    def memory(self):
        return self.session.worker.memory

    @property
    def failure(self):
        return self.session.worker.failure

    def call(self, operation, **args):
        if operation == "seed":
            self.seed = int(args["seed"])
        elif operation == "reset":
            self.image = np.asarray(args["value"]).copy()
            self.index = 0
        elif operation == "step":
            if self.seed is None or self.image is None or args.get("state_schema") is not None:
                raise ValueError("Independent image/seed required; no robot state invented")
            if args["output_views"] != ["droid.cosmos_composite"]:
                raise ValueError("Native batch only supports the declared composite view")
            future = Future()
            self.session.queue.put(({"image": self.image, "action": args["action"],
                "seed": self.seed+self.index,
                "runtime_dir": str(self.runtime_dir/f"step_{self.index:05d}")}, future))
            result = future.result()
            self.image = result["views"]["droid.cosmos_composite"][-1].copy()
            self.index += 1
            return result
        else:
            raise ValueError(f"Unsupported batched replay operation: {operation}")
        return None
