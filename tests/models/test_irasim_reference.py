"""Native selection parity is independent of CUDA, media acquisition and generation."""

import json
from pathlib import Path

import pytest

from evaluation.datasets import irasim_reference as module
from vlaworldeval.assets import file_record


def fixture(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    repo = root / "third_party/irasim"
    if not repo.exists():
        pytest.skip("Optional upstream runtime absent")
    monkeypatch.setattr(module, "COUNTS", (3, 3))
    directory = tmp_path / "annotation/test"
    directory.mkdir(parents=True)
    files, clips = {}, []
    for index, length in enumerate((15, 17, 33)):
        episode = f"{index:06d}"
        path = directory / f"{episode}.json"
        path.write_text(json.dumps({"actions": [[0.01, 0.02]] * length}))
        files[episode] = {
            "path": str(path.relative_to(tmp_path)),
            "actions": length,
            **file_record(path),
        }
        clips.extend(
            {
                "episode": episode,
                "start": start,
                "source_frame_indices": list(range(start, start + 16)),
            }
            for start in range(0, length, 16)
            if start + 15 < length
        )
    coverage = {
        "repository": "fangqi/IRASim",
        "revision": module.SOURCE_REVISION,
        "test_episodes": 3,
        "official_short_clips": 3,
        "files": files,
        "clips": clips,
    }
    (tmp_path / "coverage.json").write_text(json.dumps(coverage))
    return repo, coverage


def test_full_short_cli_rejects_a_smaller_window(tmp_path):
    from scripts.benchmarks.native_replay import main

    with pytest.raises(ValueError, match="freezes"):
        main(
            [
                "--model",
                "irasim",
                "--full-irasim-short",
                "--dataset",
                str(tmp_path),
                "--config",
                "unused",
                "--output",
                str(tmp_path / "new"),
                "--episode",
                "27",
            ]
        )
