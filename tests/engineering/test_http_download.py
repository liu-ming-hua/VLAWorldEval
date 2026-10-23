import base64
import hashlib
import io
import json

import pytest

from vlaworldeval.http_download import download_archive


def test_range_download_is_resumable_and_pinned(tmp_path, monkeypatch):
    import vlaworldeval.http_download as module

    data = b"0123456789"
    calls = []

    class Response(io.BytesIO):
        status = 206

    class Opener:
        def open(self, request, timeout):
            span = request.get_header("Range").removeprefix("bytes=")
            start, end = map(int, span.split("-"))
            calls.append((start, end))
            response = Response(data[start : end + 1])
            response.headers = {"Content-Range": f"bytes {start}-{end}/{len(data)}"}
            return response

    monkeypatch.setattr(module, "build_opener", lambda *args: Opener())
    checksum = base64.b64encode(hashlib.md5(data).digest()).decode()
    output = tmp_path / "model.zip"
    kwargs = dict(root=tmp_path, size=len(data), expected_md5=checksum, workers=2, chunk_size=3)
    assert download_archive("https://official.test/model", output, **kwargs)["ok"]
    assert output.read_bytes() == data
    assert sorted(calls) == [(0, 2), (3, 5), (6, 8), (9, 9)]
    calls.clear()
    assert download_archive("https://official.test/model", output, **kwargs)["cached"]
    assert not calls
    output.write_bytes(b"user file")
    with pytest.raises(ValueError, match="Existing archive"):
        download_archive("https://official.test/model", output, **kwargs)
    assert output.read_bytes() == b"user file"


def test_range_download_protects_unowned_parts(tmp_path):
    output = tmp_path / "model.zip"
    parts = tmp_path / "model.zip.parts"
    parts.mkdir()
    (parts / "user.txt").write_text("preserve")
    with pytest.raises(ValueError, match="unowned"):
        download_archive(
            "https://official.test/model",
            output,
            root=tmp_path,
            size=10,
            expected_md5="AAAAAAAAAAAAAAAAAAAAAA==",
        )
    assert (parts / "user.txt").read_text() == "preserve"


def test_resume_identity_symlink_is_rejected_without_network(tmp_path, monkeypatch):
    import vlaworldeval.http_download as module

    output = tmp_path / "model.zip"
    parts = tmp_path / "model.zip.parts"
    parts.mkdir()
    outside = tmp_path / "private.json"
    outside.write_text("not an identity")
    (parts / "identity.json").symlink_to(outside)
    monkeypatch.setattr(
        module, "build_opener", lambda *args: pytest.fail("No network for unsafe identity")
    )
    with pytest.raises(ValueError, match="symlink"):
        download_archive(
            "https://official.test/model",
            output,
            root=tmp_path,
            size=10,
            expected_md5="AAAAAAAAAAAAAAAAAAAAAA==",
        )
    assert outside.read_text() == "not an identity"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"workers": True},
        {"workers": 1.5},
        {"chunk_size": True},
        {"chunk_size": 0.5},
        {"size": 100000, "chunk_size": 1},
        {"expected_md5": "not-a-checksum"},
    ],
)
def test_bad_arguments_never_start_transport(tmp_path, monkeypatch, kwargs):
    import vlaworldeval.http_download as module

    monkeypatch.setattr(
        module, "build_opener", lambda *args: pytest.fail("No network for invalid input")
    )
    arguments = dict(root=tmp_path, size=10, expected_md5="AAAAAAAAAAAAAAAAAAAAAA==")
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        download_archive("https://official.test/model", tmp_path / "model.zip", **arguments)
    assert not (tmp_path / "model.zip.parts").exists()


def test_bad_content_range_cannot_publish_an_archive(tmp_path, monkeypatch):
    import vlaworldeval.http_download as module

    class Response(io.BytesIO):
        status = 200
        headers = {}

    class Opener:
        def open(self, request, timeout):
            return Response(b"0123456789")

    monkeypatch.setattr(module, "build_opener", lambda *args: Opener())
    monkeypatch.setattr(module.time, "sleep", lambda *args: None)
    output = tmp_path / "model.zip"
    with pytest.raises(ValueError, match="exact pinned byte range"):
        download_archive(
            "https://official.test/model",
            output,
            root=tmp_path,
            size=10,
            expected_md5="AAAAAAAAAAAAAAAAAAAAAA==",
        )
    assert not output.exists()


def test_fatal_range_error_does_not_queue_the_entire_download(tmp_path, monkeypatch):
    import vlaworldeval.http_download as module

    calls = []

    class Response(io.BytesIO):
        status = 200
        headers = {}

    class Opener:
        def open(self, request, timeout):
            calls.append(request.get_header("Range"))
            return Response(b"not-a-range")

    monkeypatch.setattr(module, "build_opener", lambda *args: Opener())
    monkeypatch.setattr(
        module.time, "sleep", lambda *args: pytest.fail("Do not retry a fatal protocol error")
    )
    with pytest.raises(ValueError, match="exact pinned byte range"):
        download_archive(
            "https://official.test/model",
            tmp_path / "model.zip",
            root=tmp_path,
            size=100,
            chunk_size=1,
            workers=2,
            expected_md5="AAAAAAAAAAAAAAAAAAAAAA==",
        )
    assert 1 <= len(calls) <= 2


def test_resume_repairs_same_size_corrupt_owned_piece(tmp_path, monkeypatch):
    import vlaworldeval.http_download as module

    data, calls = b"0123456789", []
    fail_second = [True]

    class Response(io.BytesIO):
        status = 206

    class Opener:
        def open(self, request, timeout):
            start, end = map(int, request.get_header("Range").removeprefix("bytes=").split("-"))
            calls.append((start, end))
            response = Response(data[start : end + 1])
            response.headers = {"Content-Range": f"bytes {start}-{end}/{len(data)}"}
            if start == 5 and fail_second[0]:
                response.status = 200
            return response

    monkeypatch.setattr(module, "build_opener", lambda *args: Opener())
    output = tmp_path / "model.zip"
    kwargs = dict(
        root=tmp_path,
        size=10,
        expected_md5=base64.b64encode(hashlib.md5(data).digest()).decode(),
        workers=1,
        chunk_size=5,
    )
    with pytest.raises(ValueError):
        download_archive("https://official.test/model", output, **kwargs)
    piece = tmp_path / "model.zip.parts/000000.part"
    assert piece.read_bytes() == b"01234"
    piece.write_bytes(b"xxxxx")  # Same size must no longer bypass cache integrity.
    fail_second[0] = False
    assert download_archive("https://official.test/model", output, **kwargs)["ok"]
    assert output.read_bytes() == data and calls.count((0, 4)) == 2


def test_legacy_piece_without_digest_is_refetched_not_blindly_trusted(tmp_path, monkeypatch):
    import vlaworldeval.http_download as module

    data = b"01234"
    expected = base64.b64encode(hashlib.md5(data).digest()).decode()
    parts = tmp_path / "model.zip.parts"
    parts.mkdir()
    (parts / "identity.json").write_text(
        json.dumps(
            {
                "url": "https://official.test/model",
                "size": 5,
                "upstream_md5": expected,
                "chunk_size": 5,
            }
        )
    )
    (parts / "000000.part").write_bytes(b"xxxxx")

    class Response(io.BytesIO):
        status = 206
        headers = {"Content-Range": "bytes 0-4/5"}

    class Opener:
        def open(self, request, timeout):
            return Response(data)

    monkeypatch.setattr(module, "build_opener", lambda *args: Opener())
    assert download_archive(
        "https://official.test/model",
        tmp_path / "model.zip",
        root=tmp_path,
        size=5,
        expected_md5=expected,
        workers=1,
        chunk_size=5,
    )["ok"]
    assert (parts / "000000.part.json").is_file()
