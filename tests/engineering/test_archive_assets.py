import base64
import hashlib
import io
import tarfile
import zipfile

import pytest

from vlaworldeval.archive import extract_archive


def checksum(path):
    return base64.b64encode(hashlib.md5(path.read_bytes()).digest()).decode()


def saved_model_zip(path, extra=None):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("saved_model.pb", b"model")
        archive.writestr("variables/variables.index", b"index")
        if extra:
            archive.writestr(extra, b"bad")


def test_verified_archive_adopts_only_matching_legacy_files(tmp_path):
    archive = tmp_path / "model.zip"
    saved_model_zip(archive)
    destination = tmp_path / "model"
    kwargs = {"root": tmp_path, "expected_md5": checksum(archive)}
    assert extract_archive(archive, destination, **kwargs)["files_verified"] == 2
    assert extract_archive(archive, destination, **kwargs)["ok"]
    (destination / "saved_model.pb").write_bytes(b"user-owned")
    with pytest.raises(ValueError, match="differs"):
        extract_archive(archive, destination, **kwargs)
    assert (destination / "saved_model.pb").read_bytes() == b"user-owned"
    assert not list(tmp_path.glob(".archive-extract-*"))


@pytest.mark.parametrize("member", ["../outside", "/absolute", "sub\\outside"])
def test_archive_rejects_unsafe_paths_without_partial_destination(tmp_path, member):
    archive = tmp_path / "model.zip"
    saved_model_zip(archive, member)
    with pytest.raises(ValueError, match="Unsafe"):
        extract_archive(archive, tmp_path / "model", root=tmp_path, expected_md5=checksum(archive))
    assert not (tmp_path / "model").exists()


def test_archive_rejects_checksum_and_symlinks(tmp_path):
    archive = tmp_path / "model.tar.gz"
    with tarfile.open(archive, "w:gz") as source:
        link = tarfile.TarInfo("saved_model.pb")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        source.addfile(link)
        data = tarfile.TarInfo("variables/variables.index")
        data.size = 5
        source.addfile(data, io.BytesIO(b"index"))
    with pytest.raises(ValueError, match="MD5"):
        extract_archive(archive, tmp_path / "model", root=tmp_path, expected_md5="bad")
    with pytest.raises(ValueError, match="links"):
        extract_archive(archive, tmp_path / "model", root=tmp_path, expected_md5=checksum(archive))
