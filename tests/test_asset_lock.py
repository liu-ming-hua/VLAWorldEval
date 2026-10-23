import os
from pathlib import Path
import select
import subprocess
import sys

import pytest

from vlaworldeval.asset_lock import asset_destination_lock
from vlaworldeval.http_download import download_archive


def test_cross_process_lock_and_kernel_release_after_exit(tmp_path):
    destination = tmp_path / "model"
    program = (
        "from pathlib import Path; import sys; "
        "from vlaworldeval.asset_lock import asset_destination_lock; "
        "lock=asset_destination_lock(Path(sys.argv[1]),Path(sys.argv[2])); "
        "lock.__enter__(); print('locked',flush=True); sys.stdin.readline()"
    )
    process = subprocess.Popen([sys.executable, "-c", program, str(tmp_path), str(destination)],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])})
    try:
        assert select.select([process.stdout], [], [], 10)[0], "Lock subprocess did not become ready"
        assert process.stdout.readline().strip() == "locked"
        with pytest.raises(ValueError, match="already running"):
            with asset_destination_lock(tmp_path, destination):
                pytest.fail("Another process must not enter this destination")
        process.terminate()
        process.wait(timeout=10)
        # A durable .lock file and stale PID must NOT prevent restart.
        assert list((tmp_path / ".cache/vlaworldeval/asset_locks").glob("*.lock"))
        with asset_destination_lock(tmp_path, destination):
            pass
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        process.stdin.close()
        process.stdout.close()
        process.stderr.close()


def test_download_refuses_concurrent_mutation_before_network(tmp_path, monkeypatch):
    import vlaworldeval.http_download as module
    monkeypatch.setattr(module, "build_opener", lambda *args: pytest.fail("No network while destination is locked"))
    output = tmp_path / "model.zip"
    with asset_destination_lock(tmp_path, output):
        with pytest.raises(ValueError, match="already running"):
            download_archive("https://official.test/model", output, root=tmp_path, size=10, expected_md5="abc")
        # Independent destinations do not block one another.
        with asset_destination_lock(tmp_path, tmp_path / "other.zip"):
            pass


def test_lock_rejects_outside_paths_and_symlink_cache(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    with pytest.raises(ValueError, match="project-local"):
        with asset_destination_lock(root, tmp_path / "outside"):
            pass
    outside = tmp_path / "outside-cache"
    outside.mkdir()
    (root / ".cache").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="leaves the project"):
        with asset_destination_lock(root, root / "model"):
            pass
    assert not list(outside.iterdir())


def test_hf_and_extraction_share_the_same_destination_lock(tmp_path):
    from vlaworldeval.archive import extract_archive
    from vlaworldeval.assets import download_snapshot
    destination = tmp_path / "model"
    with asset_destination_lock(tmp_path, destination):
        with pytest.raises(ValueError, match="already running"):
            download_snapshot("example/model", destination, root=tmp_path)
        with pytest.raises(ValueError, match="already running"):
            extract_archive(tmp_path / "not-downloaded.zip", destination,
                            root=tmp_path, expected_md5="AAAAAAAAAAAAAAAAAAAAAA==")


def test_aliases_to_same_project_directory_cannot_bypass_lock(tmp_path):
    store = tmp_path / "real-store"
    store.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(store, target_is_directory=True)
    with asset_destination_lock(tmp_path, store / "model"):
        with pytest.raises(ValueError, match="already running"):
            with asset_destination_lock(tmp_path, alias / "model"):
                pass
