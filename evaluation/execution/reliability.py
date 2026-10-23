"""Weight-free runtime settings and structured errors shared by runner/workers."""
from dataclasses import asdict, dataclass
from contextlib import contextmanager
import math
import signal


@dataclass(frozen=True)
class WorkerOptions:
    startup_timeout_seconds: float = 900
    request_timeout_seconds: float = 900
    shutdown_grace_seconds: float = 5
    terminate_grace_seconds: float = 5

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


def resolve_worker_options(runtime, role):
    if role not in {"vla", "world_model"}:
        raise ValueError(f"Unknown worker role: {role}")
    common, override = runtime.get("worker_options", {}), runtime.get(f"{role}_worker_options", {})
    if not isinstance(common, dict) or not isinstance(override, dict):
        raise ValueError("worker options must be objects")
    unknown = (set(common) | set(override)) - set(WorkerOptions.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown worker options: {sorted(unknown)}")
    return WorkerOptions(**{**common, **override})


class WorkerError(RuntimeError):
    def __init__(self, code, message, *, phase, operation=None, directory=None, returncode=None, remote=None):
        super().__init__(message)
        self.details = {"code": code, "message": message, "phase": phase, "operation": operation,
                        "directory": str(directory) if directory else None,
                        "returncode": returncode, "remote": remote}


class WorkerTimeoutError(WorkerError, TimeoutError):
    pass


class RunInterrupted(KeyboardInterrupt):
    def __init__(self, signum):
        self.signum = signum
        super().__init__(f"Run interrupted by {signal.Signals(signum).name}")


def clean_worker_exit(cleanup):
    """Require complete evidence of voluntary exit, not just launcher code zero."""
    return (isinstance(cleanup, dict) and type(cleanup.get("returncode")) is int
            and cleanup["returncode"] == 0 and cleanup.get("errors") == []
            and cleanup.get("signals") == [])


@contextmanager
def run_signal_handlers():
    """CLI cancellation unwinds finally blocks instead of orphaning workers.

    Suppress repeated cancellation while bounded cleanup is in progress, then
    restore caller-owned handlers. SIGKILL and host failure cannot be handled.
    """
    signals = (signal.SIGINT, signal.SIGTERM)
    previous = {sig: signal.getsignal(sig) for sig in signals}

    def interrupted(signum, frame):
        for sig in signals:
            signal.signal(sig, signal.SIG_IGN)
        raise RunInterrupted(signum)

    try:
        for sig in signals:
            signal.signal(sig, interrupted)
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def exception_details(exc):
    if isinstance(exc, WorkerError):
        return exc.details
    details = {"code": "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "execution_error",
               "type": type(exc).__name__, "message": str(exc)}
    if isinstance(exc, RunInterrupted):
        details["signal"] = signal.Signals(exc.signum).name
    return details


def remote_error(exc):
    # Classify explicit exceptions, not arbitrary generated text or log guesses.
    is_oom = type(exc).__name__ in {"OutOfMemoryError", "ResourceExhaustedError"}
    return {"code": "out_of_memory" if is_oom else "model_error",
            "type": type(exc).__name__, "message": str(exc)}
