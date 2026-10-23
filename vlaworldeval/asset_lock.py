"""Cross-process exclusion for project-local asset mutations.

The kernel lock, not an old PID/lock file, determines whether an operation is
live. Lock files are never unlinked: replacing their inode would allow two
independent locks on the same destination.
"""
from contextlib import contextmanager
from functools import wraps
import hashlib
import inspect
import json
import os
from pathlib import Path
import stat


@contextmanager
def asset_destination_lock(root: Path, destination: Path):
    root = Path(root).resolve()
    destination = Path(destination).absolute()
    if (destination.is_symlink() or not destination.resolve().is_relative_to(root)
            or destination.resolve() == root):
        raise ValueError("Asset destination must be project-local and not a symlink/root")
    try:
        import fcntl
    except ImportError as exc:
        raise ValueError("Automatic asset mutation requires POSIX file-lock support") from exc
    directory = root / ".cache/vlaworldeval/asset_locks"
    if not directory.resolve().is_relative_to(root):
        raise ValueError("Asset lock directory leaves the project")
    directory.mkdir(parents=True, exist_ok=True)
    identity = str(destination.resolve())
    path = directory / (hashlib.sha256(identity.encode()).hexdigest() + ".lock")
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NONBLOCK | os.O_NOFOLLOW, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("Asset lock must be a regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Asset operation already running for {destination}; retry after it exits") from exc
        record = json.dumps({"pid": os.getpid(), "destination": identity}).encode()
        os.ftruncate(descriptor, 0)
        os.write(descriptor, record)
        yield
    finally:
        os.close(descriptor)  # Also releases locks after errors; crashes release them in the kernel.


def destination_locked(parameter):
    """Preserve the public function signature while locking its destination."""
    def decorate(function):
        signature = inspect.signature(function)

        @wraps(function)
        def wrapped(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            with asset_destination_lock(bound.arguments["root"], bound.arguments[parameter]):
                return function(*args, **kwargs)
        return wrapped
    return decorate
