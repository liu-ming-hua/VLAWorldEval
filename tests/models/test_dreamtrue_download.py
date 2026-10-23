"""Small CPU asset-integrity checks; never download or load model tensors."""

import hashlib
import importlib.util
import io
import tarfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "dreamtrue_download", Path(__file__).resolve().parents[2] / "scripts/download/dreamtrue.py"
)
download = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download)


def test_immutable_release_and_shared_hf_acquisition(monkeypatch, tmp_path):
    assert len(download.DATASET_REVISION) == 40
    monkeypatch.setattr(download, "ROOT", tmp_path)
    calls = []
    monkeypatch.setattr(
        download, "download_snapshot", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    download.download_base()
    args, kwargs = calls[0]
    assert args[0] == "Wan-AI/Wan2.1-VACE-14B"
    assert kwargs["revision"] == download.BASE_REVISION
    assert "diffusion_pytorch_model*" in kwargs["patterns"]


@pytest.mark.parametrize(
    "name,type_",
    [
        ("../outside", tarfile.REGTYPE),
        ("/outside", tarfile.REGTYPE),
        ("link", tarfile.SYMTYPE),
        ("fifo", tarfile.FIFOTYPE),
    ],
)
def test_extraction_rejects_unsafe_members(tmp_path, name, type_):
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        info = tarfile.TarInfo(name)
        info.type = type_
        bundle.addfile(info)
    with pytest.raises(ValueError, match="Unsafe"):
        download.extract_conditions(archive, tmp_path / "release")


def test_extraction_preserves_existing_files(tmp_path):
    archive = tmp_path / "safe.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        info = tarfile.TarInfo("condition/config.yaml")
        info.size = 2
        bundle.addfile(info, io.BytesIO(b"ok"))
    target = tmp_path / "release"
    download.extract_conditions(archive, target)
    assert (target / "condition/config.yaml").read_bytes() == b"ok"
    with pytest.raises(FileExistsError):
        download.extract_conditions(archive, target)


def test_existing_bad_asset_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "ROOT", tmp_path)
    path = tmp_path / "asset"
    path.write_bytes(b"user data")
    with pytest.raises(ValueError, match="not overwrite"):
        download.fetch("https://unused.invalid", path, 1, "0" * 64)
    assert path.read_bytes() == b"user data"


def test_native_adapter_does_not_depend_on_scripts_package():
    text = (Path(__file__).resolve().parents[2] / "adapters/world_models/dreamtrue.py").read_text()
    assert "from scripts." not in text


def test_verified_range_resume_preserves_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "ROOT", tmp_path)
    target = tmp_path / "asset"
    target.with_name("asset.partial").write_bytes(b"abc")
    requests = []

    class Response(io.BytesIO):
        status = 206
        headers = {"Content-Range": "bytes 3-5/6"}

    def open_url(request, **kwargs):
        requests.append(request)
        return Response(b"def")

    monkeypatch.setattr(download.urllib.request, "urlopen", open_url)
    download.fetch("https://unused.invalid", target, 6, hashlib.sha256(b"abcdef").hexdigest())
    assert target.read_bytes() == b"abcdef"
    assert requests[0].get_header("Range") == "bytes=3-5"


def test_two_writers_cannot_share_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "ROOT", tmp_path)
    target = tmp_path / "asset"
    with download.asset_destination_lock(tmp_path, target):
        with pytest.raises(ValueError, match="already running"):
            download.fetch("https://unused.invalid", target, 0, hashlib.sha256(b"").hexdigest())


def test_multiple_bounded_ranges_are_appended_before_hash_admission(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "ROOT", tmp_path)
    monkeypatch.setattr(download, "DOWNLOAD_RANGE_BYTES", 3)
    content = b"abcdefgh"
    requests = []

    class Response(io.BytesIO):
        status = 206

    def open_url(request, **kwargs):
        value = request.get_header("Range")
        requests.append(value)
        start, end = map(int, value.removeprefix("bytes=").split("-"))
        result = Response(content[start : end + 1])
        result.headers = {"Content-Range": f"bytes {start}-{end}/{len(content)}"}
        return result

    monkeypatch.setattr(download.urllib.request, "urlopen", open_url)
    target = tmp_path / "asset"
    download.fetch(
        "https://unused.invalid", target, len(content), hashlib.sha256(content).hexdigest()
    )
    assert target.read_bytes() == content
    assert requests == ["bytes=0-2", "bytes=3-5", "bytes=6-7"]


def test_partial_symlink_cannot_escape_project(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "ROOT", tmp_path)
    original = tmp_path / "original"
    original.write_bytes(b"user data")
    (tmp_path / "asset.partial").symlink_to(original)
    with pytest.raises(ValueError, match="symlink"):
        download.fetch("https://unused.invalid", tmp_path / "asset", 5, "0" * 64)
    assert original.read_bytes() == b"user data"
