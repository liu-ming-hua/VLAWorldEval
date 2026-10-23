"""Isolated bounded-request shim for the pinned gdown 6.2.0 downloader."""
from importlib.metadata import version
import math
from pathlib import Path
import types


def download_google_drive(file_id: str, destination: Path, *, connect_timeout=30, read_timeout=600):
    for value in (connect_timeout, read_timeout):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("Google Drive connect/read timeouts must be finite and positive")
    try:
        import gdown
    except ImportError as exc:
        raise ValueError("Download requires gdown==6.2.0 in the selected tool environment") from exc
    if version("gdown") != "6.2.0":
        raise ValueError("Bounded Google Drive profile requires gdown==6.2.0; install the pinned version")
    original = gdown.download
    factory = original.__globals__["_get_session"]
    timeout = (float(connect_timeout), float(read_timeout))

    def bounded_session(**kwargs):
        session, cookies = factory(**kwargs)
        session.trust_env = False  # Public checkpoint: no ambient proxies or netrc credentials.
        request = session.request

        def bounded_request(method, url, **options):
            options.setdefault("timeout", timeout)
            return request(method, url, **options)

        session.request = bounded_request
        return session, cookies

    # gdown exposes no timeout argument. Use a private copy of the pinned
    # function's globals instead of monkey-patching its module or requests;
    # simultaneous downloads for other destinations keep their own sessions.
    namespace = dict(original.__globals__)
    namespace["_get_session"] = bounded_session
    bounded = types.FunctionType(original.__code__, namespace, original.__name__,
                                 original.__defaults__, original.__closure__)
    bounded.__kwdefaults__ = original.__kwdefaults__
    return bounded(id=file_id, output=str(destination), resume=True, use_cookies=False,
                   proxy=None, verify=True)
