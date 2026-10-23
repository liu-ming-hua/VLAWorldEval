"""Persistent subprocess runtimes exposed as ordinary evaluator adapters."""
from __future__ import annotations

import json
from dataclasses import asdict
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time

from adapters.base import VLAAdapter, WorldModelAdapter
from evaluation.contracts import RobotState, WorldModelOutput
from evaluation.execution.transport import read_packet, write_packet
from evaluation.execution.reliability import WorkerError, WorkerOptions, WorkerTimeoutError, clean_worker_exit

from vlaworldeval.paths import project_root
from vlaworldeval.cache import runtime_cache_environment

ROOT = project_root()


def worker_code_path(directory, code_root):
    """Do not put an installed core's entire site-packages on model PYTHONPATH."""
    if (code_root / "pyproject.toml").is_file():
        return code_root
    stage = directory / "framework_code"
    stage.mkdir()
    for name in ("adapters", "cli", "evaluation", "vlaworldeval"):
        shutil.copytree(code_root / name, stage / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return stage


class ModelWorker:
    def __init__(self, python: Path, directory: Path, *, module: str, adapter: str,
                 kwargs: dict, gpus: list[int], memory_gib: float, timeout: float | None = None,
                 options: WorkerOptions | None = None):
        if timeout is not None and options is not None:
            raise ValueError("Use options or legacy timeout, not both")
        self.options = options or (WorkerOptions(startup_timeout_seconds=timeout, request_timeout_seconds=timeout)
                                   if timeout is not None else WorkerOptions())
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.process = self.log = None
        self.closed = False
        self.failure = None
        self.cleanup = None
        self.counter = 0
        self.memory = None
        config = {"module": module, "adapter": adapter, "kwargs": kwargs,
                  "gpus": gpus, "memory_gib": memory_gib, "worker_options": asdict(self.options)}
        env = os.environ.copy()
        cache_environment = runtime_cache_environment(ROOT)
        env.update(cache_environment)
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            env.pop(key, None)
        installed_root = Path(__file__).resolve().parents[2]
        code_root = worker_code_path(self.directory, installed_root)
        config["framework_code_root"] = str(code_root)
        env.update(PYTHONPATH=str(code_root), VLAWORLDEVAL_ROOT=str(ROOT), CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)),
                   HF_HOME=str(ROOT / ".cache/huggingface"), HF_HUB_OFFLINE="1",
                   TRANSFORMERS_OFFLINE="1", NO_ALBUMENTATIONS_UPDATE="1",
                   USE_TF="0",
                   XLA_PYTHON_CLIENT_PREALLOCATE="false", PYTHONUNBUFFERED="1")
        config["environment"] = {key: env[key] for key in (*cache_environment,
            "CUDA_VISIBLE_DEVICES", "HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
            "USE_TF", "XLA_PYTHON_CLIENT_PREALLOCATE",
        )}
        config["proxies_removed"] = True
        (self.directory / "worker.json").write_text(json.dumps(config, indent=2))
        command = [str(python)]
        if len(gpus) > 1:
            command += ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={len(gpus)}"]
        command += ["-m", "evaluation.execution.worker", "--directory", str(self.directory)]
        self.log = (self.directory / "console.log").open("w")
        try:
            try:
                self.process = subprocess.Popen(command, cwd=ROOT if code_root == installed_root else self.directory,
                                                env=env, stdin=subprocess.PIPE,
                                                stdout=self.log, stderr=subprocess.STDOUT, text=True, start_new_session=True)
            except OSError as exc:
                raise WorkerError("launch_failed", str(exc), phase="startup", directory=self.directory) from exc
            self._lifecycle("starting")
            self.ready = self._wait(self.directory / "ready.npz", phase="startup", operation=None)
            if self.ready["adapter"] != adapter:
                raise WorkerError("identity_mismatch", "Worker adapter identity mismatch", phase="startup",
                                  directory=self.directory)
            self._lifecycle("ready")
        except BaseException as exc:
            self._fail(exc)
            raise

    def _write_json(self, name, value):
        path = self.directory / name
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
        temporary.replace(path)

    def _lifecycle(self, status):
        self._write_json("lifecycle.json", {"status": status, "pid": self.process.pid if self.process else None,
                                          "request_count": self.counter, "error": self.failure,
                                          "cleanup": self.cleanup})

    def _fail(self, exc):
        from evaluation.execution.reliability import exception_details
        self.failure = exception_details(exc)
        try:
            self._write_json("error.json", self.failure)
        except OSError:
            pass  # Preserve the original failure even if disk is unavailable.
        self.close(force=True)

    def _wait(self, path, *, phase, operation):
        seconds = (self.options.startup_timeout_seconds if phase == "startup"
                   else self.options.request_timeout_seconds)
        deadline = time.monotonic() + seconds
        while not path.is_file():
            if self.process.poll() is not None:
                raise WorkerError("worker_exited", f"Worker exited ({self.process.returncode}); see {self.directory / 'console.log'}",
                                  phase=phase, operation=operation, directory=self.directory,
                                  returncode=self.process.returncode)
            if time.monotonic() >= deadline:
                raise WorkerTimeoutError(f"{phase}_timeout", f"Worker {phase} timeout after {seconds}s: {path}",
                                         phase=phase, operation=operation, directory=self.directory)
            time.sleep(min(0.05, max(0, deadline - time.monotonic())))
        try:
            value = read_packet(path)
            if not isinstance(value, dict) or type(value.get("ok")) is not bool:
                raise ValueError("Missing boolean ok in worker packet")
            if value["ok"] and ((phase == "startup" and "adapter" not in value)
                                or (phase == "request" and "result" not in value)):
                raise ValueError("Incomplete successful worker packet")
            if not value["ok"] and (not isinstance(value.get("error"), str)
                                    or not isinstance(value.get("error_info", {}), dict)):
                raise ValueError("Invalid worker failure packet")
        except Exception as exc:
            raise WorkerError("protocol_error", str(exc), phase=phase, operation=operation,
                              directory=self.directory) from exc
        if not value["ok"]:
            remote = value.get("error_info", {})
            code = "out_of_memory" if remote.get("code") == "out_of_memory" else (
                "startup_failed" if phase == "startup" else "model_error")
            raise WorkerError(code, value.get("error", "Worker reported failure"), phase=phase,
                              operation=operation, directory=self.directory, remote=remote)
        return value

    def call(self, operation, **args):
        if self.closed:
            raise WorkerError("worker_unavailable", "Worker is closed; failed workers cannot be reused",
                              phase="request", operation=operation, directory=self.directory)
        self.counter += 1
        name = f"request_{self.counter:06d}.npz"
        request, response = self.directory / name, self.directory / name.replace("request_", "response_")
        completed = False
        try:
            try:
                write_packet(request, {"operation": operation, "args": args})
            except Exception as exc:
                raise WorkerError("transport_error", str(exc), phase="request", operation=operation,
                                  directory=self.directory) from exc
            try:
                self.process.stdin.write(name + "\n")
                self.process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise WorkerError("worker_exited", str(exc), phase="request", operation=operation,
                                  directory=self.directory, returncode=self.process.poll()) from exc
            value = self._wait(response, phase="request", operation=operation)
            self.memory = value.get("memory")
            completed = True
            return value["result"]
        except BaseException as exc:
            self._fail(exc)
            raise
        finally:
            if completed:
                request.unlink(missing_ok=True)
                response.unlink(missing_ok=True)

    def close(self, *, force=False):
        """Idempotently close only this session's process group, including children."""
        if self.closed:
            return self.cleanup
        self.closed = True
        self.cleanup = {"signals": [], "errors": [], "returncode": None}
        process = self.process

        def group_alive():
            try:
                os.killpg(process.pid, 0)
                return True
            except ProcessLookupError:
                return False

        def send(sig):
            try:
                os.killpg(process.pid, sig)
                self.cleanup["signals"].append(signal.Signals(sig).name)
            except ProcessLookupError:
                pass

        try:
            if process is not None:
                if not force and process.poll() is None:
                    try:
                        process.stdin.write("close\n")
                        process.stdin.flush()
                        process.wait(timeout=self.options.shutdown_grace_seconds)
                    except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                        pass
                # A dead launcher may leave ranks/children alive in its group.
                if group_alive():
                    send(signal.SIGTERM)
                    deadline = time.monotonic() + self.options.terminate_grace_seconds
                    while group_alive() and time.monotonic() < deadline:
                        process.poll()  # Reap launcher before checking group again.
                        time.sleep(0.02)
                    if group_alive():
                        send(signal.SIGKILL)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.cleanup["errors"].append("Launcher did not exit after SIGKILL")
                self.cleanup["returncode"] = process.poll()
        except OSError as exc:
            self.cleanup["errors"].append(str(exc))
        finally:
            for handle in (process.stdin if process else None, self.log):
                if handle is not None:
                    try:
                        handle.close()
                    except OSError as exc:
                        self.cleanup["errors"].append(str(exc))
            try:
                if self.failure is None and not clean_worker_exit(self.cleanup):
                    error = WorkerError("shutdown_failed", "Worker did not exit cleanly after close",
                                        phase="cleanup", directory=self.directory,
                                        returncode=self.cleanup["returncode"])
                    self.failure = error.details
                    self._write_json("error.json", self.failure)
                self._lifecycle("failed" if self.failure else "closed")
            except OSError as exc:
                self.cleanup["errors"].append(str(exc))
        return self.cleanup


class RemoteVLA(VLAAdapter):
    def __init__(self, worker: ModelWorker):
        self.worker = worker
        self.adapter_identity = worker.ready["adapter"]

    def reset(self, initial_state=None):
        self.worker.call("reset", value=initial_state)

    def predict(self, image, instruction, state=None):
        return self.worker.call("predict", image=image, instruction=instruction, state=state)


class RemoteWorldModel(WorldModelAdapter):
    def __init__(self, worker: ModelWorker):
        self.worker = worker
        self.adapter_identity = worker.ready["adapter"]

    def reset(self, image):
        self.worker.call("reset", value=image)

    def step_output(self, action, *, output_views=(), state_schema=None):
        result = self.worker.call("step", action=action, output_views=list(output_views), state_schema=state_schema)
        state = result["robot_state"]
        return WorldModelOutput(views=result["views"], robot_state=RobotState(**state) if state else None,
                                metadata=result["metadata"])

    def step(self, action):
        raise RuntimeError("Remote world models require the protocol-aware step_output API")
