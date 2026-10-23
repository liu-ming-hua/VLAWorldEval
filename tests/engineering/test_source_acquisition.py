"""Offline Git transactions: failed acquisitions must not poison configured paths."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.process


HELPER = Path(__file__).resolve().parents[2] / "scripts/setup/common.sh"


def git(directory, *args):
    return subprocess.check_output(["git", "-C", str(directory), *args], text=True).strip()


@pytest.fixture
def upstream(tmp_path):
    repo = tmp_path / "remote"
    repo.mkdir()
    git(repo, "init", "-q")
    git(
        repo,
        "-c",
        "user.name=fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        "first",
    )
    return repo, git(repo, "rev-parse", "HEAD")


def acquire(source, remote, revision, *, env=None):
    # Intentionally use an if-condition: Bash disables errexit inside a
    # function called this way. The helper must check every Git operation itself.
    return subprocess.run(
        [
            "bash",
            "-c",
            'set -eu; source "$1"; if ensure_source "$2" "$3" "$4"; then exit 0; else exit 7; fi',
            "fixture",
            str(HELPER),
            str(source),
            str(remote),
            revision,
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )


@pytest.mark.parametrize("name", ["model", "model with spaces"])
def test_success_publishes_only_pinned_complete_checkout(tmp_path, upstream, name):
    remote, revision = upstream
    source = tmp_path / "third_party" / name
    result = acquire(source, remote, revision)
    assert result.returncode == 0, result.stderr
    assert git(source, "rev-parse", "HEAD") == revision
    assert not list(source.parent.glob(".source-fetch.*"))


@pytest.mark.parametrize("revision", ["main", "123abcd"])
def test_mutable_or_short_revision_is_rejected_before_clone(tmp_path, upstream, revision):
    remote, _ = upstream
    source = tmp_path / "model"
    result = acquire(source, remote, revision)
    assert result.returncode == 7 and "exact 40-character" in result.stderr
    assert not source.exists()
    assert not list(tmp_path.glob(".source-fetch.*"))


def test_failed_checkout_is_retained_outside_formal_path_and_retry_works(tmp_path, upstream):
    remote, revision = upstream
    source = tmp_path / "third_party/model"
    failed = acquire(source, remote, "0" * 40)
    assert failed.returncode == 7
    assert not source.exists()
    stages = list(source.parent.glob(".source-fetch.*"))
    assert len(stages) == 1 and (stages[0] / "checkout/.git").is_dir()
    assert str(stages[0]) in failed.stderr
    assert acquire(source, remote, revision).returncode == 0
    assert git(source, "rev-parse", "HEAD") == revision
    assert stages[0].is_dir()  # A retry never deletes previous failed material.


def test_failed_clone_leaves_formal_path_absent(tmp_path):
    source = tmp_path / "third_party/model"
    result = acquire(source, tmp_path / "missing-remote", "a" * 40)
    assert result.returncode == 7 and not source.exists()
    assert "retained staging" in result.stderr


def test_existing_user_directory_is_untouched(tmp_path, upstream):
    remote, revision = upstream
    source = tmp_path / "user-model"
    source.mkdir()
    marker = source / "important.py"
    marker.write_text("user work")
    result = acquire(source, remote, revision)
    assert result.returncode == 7 and "not deleting" in result.stderr
    assert marker.read_text() == "user work"
    assert not list(tmp_path.glob(".source-fetch.*"))


def test_existing_pinned_modified_checkout_is_preserved(tmp_path, upstream):
    remote, revision = upstream
    source = tmp_path / "model"
    assert acquire(source, remote, revision).returncode == 0
    marker = source / "local_patch.py"
    marker.write_text("intentional local patch")
    result = acquire(source, remote, revision)
    assert result.returncode == 0 and "preserving local" in result.stderr
    assert marker.read_text() == "intentional local patch"


def test_existing_other_revision_is_never_reset(tmp_path, upstream):
    remote, revision = upstream
    source = tmp_path / "model"
    assert acquire(source, remote, revision).returncode == 0
    result = acquire(source, remote, "a" * 40)
    assert result.returncode == 7 and "revision mismatch" in result.stderr
    assert git(source, "rev-parse", "HEAD") == revision


@pytest.mark.parametrize("mode", ["submodule_failure", "concurrent_destination"])
def test_late_failure_does_not_publish_or_overwrite(tmp_path, upstream, mode):
    remote, revision = upstream
    source = tmp_path / "model"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "git"
    stub.write_text("""#!/usr/bin/env bash
if [[ "$*" == *"submodule update"* ]]; then
  if [[ "$ACQUISITION_TEST_MODE" == "submodule_failure" ]]; then exit 9; fi
  mkdir "$ACQUISITION_TEST_TARGET"
  printf "foreign work" > "$ACQUISITION_TEST_TARGET/marker"
fi
exec "$ACQUISITION_REAL_GIT" "$@"
""")
    stub.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        "ACQUISITION_REAL_GIT": shutil.which("git"),
        "ACQUISITION_TEST_MODE": mode,
        "ACQUISITION_TEST_TARGET": str(source),
    }
    result = acquire(source, remote, revision, env=env)
    assert result.returncode == 7
    assert len(list(tmp_path.glob(".source-fetch.*"))) == 1
    if mode == "submodule_failure":
        assert not source.exists()
    else:
        assert (source / "marker").read_text() == "foreign work"
        assert not (source / ".git").exists()
