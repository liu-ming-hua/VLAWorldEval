"""Transport retry is not authorization to replace corrupt files or bypass gates."""

from types import SimpleNamespace

import pytest

from scripts.download import native_evaluation as module
from vlaworldeval import assets


def test_transport_resume_has_one_dataset_and_preserves_attempt_evidence(monkeypatch):
    calls, snapshots = [], []

    def fetch():
        calls.append(True)
        if len(calls) == 1:
            raise ConnectionError("fixture interrupted body")
        return {"ok": True, "revision": "a" * 40}

    monkeypatch.setattr(module.time, "sleep", lambda delay: None)
    entry = {}
    module.download_with_retry(fetch, entry, lambda: snapshots.append(entry["status"]))
    assert len(calls) == len(entry["attempts"]) == 2
    assert entry["status"] == "verified" and "retrying_transport" in snapshots
    assert entry["attempts"][0]["transport_retryable"] is True
    assert entry["substitute_used"] is False


@pytest.mark.parametrize(
    "failure", [ValueError("existing file differs"), PermissionError("fixture denied")]
)
def test_non_transport_errors_are_not_retried(failure):
    entry, calls = {}, []

    def fetch():
        calls.append(True)
        raise failure

    assert module.download_with_retry(fetch, entry, lambda: None) is None
    assert entry["status"] == "unavailable" and len(calls) == 1


def test_gated_http403_and_transient_http503_are_distinct():
    error = RuntimeError("fixture HTTP")
    error.response = SimpleNamespace(status_code=403)
    assert not module.retryable_transport_error(error)
    error.response.status_code = 503
    assert module.retryable_transport_error(error)


def test_exhausted_network_attempts_are_not_called_missing_dataset(monkeypatch):
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    entry = {}

    def fetch():
        raise TimeoutError("fixture timed out")

    module.download_with_retry(fetch, entry, lambda: None, attempts=2)
    assert entry["status"] == "transport_failed" and len(entry["attempts"]) == 2


def test_cancel_during_backoff_is_recorded(monkeypatch):
    def cancel(_):
        raise KeyboardInterrupt()

    monkeypatch.setattr(module.time, "sleep", cancel)
    entry = {}
    with pytest.raises(KeyboardInterrupt):
        module.download_with_retry(
            lambda: (_ for _ in ()).throw(ConnectionError()), entry, lambda: None
        )
    assert entry["status"] == "interrupted"


def test_per_file_resume_does_not_repeat_the_dataset(monkeypatch):
    calls = []
    monkeypatch.setattr(assets, "time", SimpleNamespace(sleep=lambda _: None))

    def get_same_file():
        calls.append("immutable-part-aa")
        if len(calls) < 3:
            raise ConnectionError("fixture partial body")
        return "complete"

    assert assets.transfer_file(get_same_file, filename="part-aa") == "complete"
    assert calls == ["immutable-part-aa"] * 3


def test_per_file_integrity_failure_is_not_a_retry(monkeypatch):
    calls = []

    def get_same_file():
        calls.append(True)
        raise ValueError("fixture checksum differs")

    with pytest.raises(ValueError):
        assets.transfer_file(get_same_file, filename="part-aa")
    assert calls == [True]


def test_hub_local_cache_wrapper_preserves_actual_network_cause(monkeypatch):
    class WrappedLocalEntryNotFoundError(FileNotFoundError, ValueError):
        pass

    error = WrappedLocalEntryNotFoundError("fixture Hub could not populate cache")
    error.__cause__ = ConnectionError("fixture network unavailable")
    assert module.retryable_transport_error(error)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    entry = {}
    module.download_with_retry(
        lambda: (_ for _ in ()).throw(error), entry, lambda: None, attempts=2
    )
    assert entry["status"] == "transport_failed" and len(entry["attempts"]) == 2


@pytest.mark.parametrize("status", [401, 403, 404])
def test_authorization_or_missing_http_status_overrides_network_context(status):
    denied = RuntimeError("fixture HTTP response")
    denied.response = SimpleNamespace(status_code=status)
    denied.__context__ = ConnectionError("older unrelated network error")
    wrapper = RuntimeError("fixture Hub wrapper")
    wrapper.__cause__ = denied
    assert not module.retryable_transport_error(wrapper)


def test_exception_chain_cycles_do_not_loop_forever():
    first, second = RuntimeError("first"), RuntimeError("second")
    first.__cause__, second.__cause__ = second, first
    assert not module.retryable_transport_error(first)
