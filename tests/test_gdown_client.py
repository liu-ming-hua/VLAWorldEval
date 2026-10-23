import sys
import types

import pytest

from vlaworldeval.gdown_client import download_google_drive


def download_stub(*, id, output, resume, use_cookies, proxy, verify):
    session, _ = _get_session(proxy=proxy, use_cookies=use_cookies, user_agent="fixture", cookies_file=None)  # noqa: F821
    session.get("https://drive.google.com/fixture", verify=verify)
    return {"output": output, "id": id, "resume": resume}


def test_bounded_gdown_uses_local_sessions_without_global_monkeypatch(tmp_path, monkeypatch):
    import vlaworldeval.gdown_client as module
    calls, factories = [], []
    class Session:
        def request(self, method, url, **kwargs):
            kwargs["trust_env"] = getattr(self, "trust_env", True)
            calls.append(kwargs)
        def get(self, url, **kwargs):
            return self.request("GET", url, **kwargs)
    def factory(**kwargs):
        factories.append(kwargs)
        return Session(), "unused-cookie-file"
    original = types.FunctionType(download_stub.__code__, {"_get_session": factory})
    library = types.SimpleNamespace(download=original)
    monkeypatch.setitem(sys.modules, "gdown", library)
    monkeypatch.setattr(module, "version", lambda name: "6.2.0")
    result = download_google_drive("fixture", tmp_path / "model", connect_timeout=3, read_timeout=7)
    assert result["resume"]
    assert calls[0] == {"verify": True, "timeout": (3, 7), "trust_env": False}
    assert factories[0]["use_cookies"] is False and factories[0]["proxy"] is None
    assert library.download is original and original.__globals__["_get_session"] is factory
    # An unrelated call to the original library must NOT inherit the shim.
    original(id="fixture", output="other", resume=True, use_cookies=False, proxy=None, verify=True)
    assert "timeout" not in calls[1]


def test_unknown_gdown_version_fails_before_download(tmp_path, monkeypatch):
    import vlaworldeval.gdown_client as module
    monkeypatch.setitem(sys.modules, "gdown", types.SimpleNamespace())
    monkeypatch.setattr(module, "version", lambda name: "future-unverified")
    with pytest.raises(ValueError, match="pinned version"):
        download_google_drive("fixture", tmp_path / "model")


@pytest.mark.parametrize("value", [True, 0, -1, float("inf"), float("nan")])
def test_timeout_validation_happens_before_import_or_io(tmp_path, value):
    with pytest.raises(ValueError, match="finite and positive"):
        download_google_drive("fixture", tmp_path / "model", read_timeout=value)
