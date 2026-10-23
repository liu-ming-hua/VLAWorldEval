"""One resident worker across cases, with deferred clean-exit certification."""
from dataclasses import asdict
import json
from pathlib import Path

from evaluation.execution.client import ModelWorker
from evaluation.execution.reliability import clean_worker_exit
from evaluation.reporting.task_scores import _digest
from vlaworldeval.atomic_json import publish_json_exclusive, replace_owned_json


class NativeWorkerSession:
    def __init__(self, directory, *, stop_path=None, on_generated=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.worker = None
        self.identity = None
        self.reports = {}
        self.studies = set()
        self.closed = False
        self.cases = 0
        self.stop_path = Path(stop_path) if stop_path is not None else None
        self.clean = False
        self.on_generated = on_generated

    @property
    def stop_requested(self):
        return self.stop_path is not None and self.stop_path.exists()

    def owns_resources(self, gpus, memory_gib):
        """A live leased worker already reserved this allocator budget.

        Re-admitting the entire budget against remaining free VRAM would count
        our own resident allocations twice. Identity is checked again at acquire.
        """
        if self.worker is None or self.worker.closed or self.worker.process.poll() is not None:
            return False
        identity = json.loads(self.identity)
        return identity["gpus"] == gpus and identity["memory_gib"] == memory_gib

    def acquire(self, python, *, module, adapter, kwargs, gpus, memory_gib, options):
        if self.closed:
            raise ValueError("Resident session already closed")
        if "runtime_dir" not in kwargs:
            raise ValueError("Resident native reuse needs an explicit episode runtime_dir")
        identity = json.dumps({"python": str(Path(python).resolve()), "module": module,
            "adapter": adapter, "kwargs": {k: v for k, v in kwargs.items() if k != "runtime_dir"},
            "gpus": gpus, "memory_gib": memory_gib, "options": asdict(options)}, sort_keys=True)
        reused = self.worker is not None
        if reused and identity != self.identity:
            raise ValueError("Resident worker cannot change checkpoint, scientific config, runtime or GPUs")
        if not reused:
            self.identity = identity
            self.worker = ModelWorker(python, self.directory/"worker", module=module, adapter=adapter,
                kwargs=kwargs, gpus=gpus, memory_gib=memory_gib, options=options)
            if not self.worker.ready.get("supports_episode_reuse", False):
                self.worker.close()
                raise ValueError("Adapter does not declare verified episode reuse")
        self.worker.call("episode_context", runtime_dir=kwargs["runtime_dir"])
        self.cases += 1
        return self.worker, reused

    def track_report(self, path):
        path = Path(path)
        self.reports[path] = _digest(path)
        if self.on_generated is not None and json.loads(path.read_text())["status"] == "generated_pending_worker_close":
            self.on_generated(path)

    def track_study(self, directory):
        self.studies.add(Path(directory))

    def close(self):
        if self.closed:
            return
        self.closed = True
        cleanup = self.worker.close() if self.worker is not None else None
        clean = self.worker is None or (not self.worker.failure and clean_worker_exit(cleanup))
        self.clean = clean
        evidence = {"schema_version": 1, "status": "closed" if clean else "failed",
            "cases": self.cases, "model_loads": int(self.worker is not None), "cleanup": cleanup,
            "worker_directory": str(self.directory/"worker"), "model_identity": self.identity}
        publish_json_exclusive(self.directory/"session_close.json", evidence)
        for path, digest in self.reports.items():
            if _digest(path) != digest:
                raise ValueError("Generated case report changed before session certification")
            report = json.loads(path.read_text())
            report["cleanup"] = cleanup
            report["worker_session"]["close_manifest"] = str(self.directory/"session_close.json")
            report["worker_session"]["close_manifest_sha256"] = _digest(self.directory/"session_close.json")
            if report["status"] == "generated_pending_worker_close":
                report["status"] = "completed" if clean else "failed"
            if not clean:
                report.update(status="failed", error={"code": "resident_worker_cleanup_failed"})
            replace_owned_json(path, report)
        for directory in self.studies:
            path = directory/"progress.json"
            if not path.exists():
                continue
            progress = json.loads(path.read_text())
            for identifier, entry in progress["jobs"].items():
                report_path = directory/"runs"/identifier/"summary.json"
                if report_path not in self.reports:
                    continue
                report = json.loads(report_path.read_text())
                entry.update(status=report["status"], report_sha256=_digest(report_path))
            study = json.loads((directory/"study.json").read_text())
            statuses = [j["status"] for j in progress["jobs"].values()]
            progress["status"] = ("completed" if len(statuses) == len(study["jobs"])
                                  and all(s == "completed" for s in statuses)
                                  else "failed" if any(s == "failed" for s in statuses) else "stopped_partial")
            replace_owned_json(path, progress)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
