"""Optional native-domain parity; requires separately acquired assets or runtimes."""

import json
import shutil
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


def test_exact_upstream_indexing_accounts_for_episodes_with_no_short_window(tmp_path, monkeypatch):
    repo, _ = fixture(tmp_path, monkeypatch)
    result = module.short_selection(tmp_path, repo)
    assert result["all_indices_equal"]
    assert result["episodes_without_short_window"] == ["000000"]
    assert [(row["episode"], row["start"]) for row in result["samples"]] == [
        ("000001", 0),
        ("000002", 0),
        ("000002", 16),
    ]
    assert result["model_inference_performed"] is result["full_test_set_completed"] is False


@pytest.mark.parametrize("problem", ["bytes", "window", "count", "path"])
def test_short_selection_never_falls_back_on_changed_or_missing_data(
    tmp_path, monkeypatch, problem
):
    repo, coverage = fixture(tmp_path, monkeypatch)
    if problem == "bytes":
        (tmp_path / "annotation/test/000001.json").write_text("{}")
    elif problem == "window":
        coverage["clips"][0]["source_frame_indices"] = list(range(1, 17))
    elif problem == "count":
        coverage["test_episodes"] = 2
    else:
        coverage["files"]["000001"]["path"] = "../outside.json"
    (tmp_path / "coverage.json").write_text(json.dumps(coverage))
    with pytest.raises(ValueError):
        module.short_selection(tmp_path, repo)


@pytest.mark.parametrize("mode", ["short", "long"])
def test_full_preparation_freezes_all_cases_and_resumes_checked_cases(tmp_path, monkeypatch, mode):
    annotations = tmp_path / "annotations"
    repo, coverage = fixture(annotations, monkeypatch)
    source = Path(__file__).resolve().parents[2]
    for relative in (
        "evaluation/datasets/irasim_reference.py",
        "evaluation/execution/transport.py",
        "evaluation/manifests.py",
        "evaluation/timeline.py",
        "third_party/irasim/dataset/dataset_2D.py",
        "third_party/irasim/dataset/video_transforms.py",
        "third_party/irasim/evaluate/generate_long_video.py",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
    assets = tmp_path / "assets"
    records = {}
    for episode in coverage["files"]:
        for name in (
            f"annotation/test/{episode}.json",
            f"videos/test/{episode}/rgb.mp4",
            f"latent_videos/test/{episode}/0.pt",
        ):
            path = assets / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture bytes; never sent to a real decoder")
            records[name] = file_record(path)
    from evaluation.reporting.task_scores import _digest

    (assets / "coverage.json").write_text(
        json.dumps(
            {
                "status": "extracted_not_inferred",
                "test_episodes": 3,
                "official_short_clips": 3,
                "annotation_manifest_sha256": _digest(annotations / "coverage.json"),
                "source_snapshot": {"revision": module.SOURCE_REVISION},
                "files": records,
            }
        )
    )
    config = tmp_path / "config.json"
    shutil.copyfile(source / "configs/replay/irasim_native_v1.json", config)
    monkeypatch.setattr(module, "load_episode", lambda *_: "fixture only")
    calls = []

    def write_case(dataset, upstream, runtime, target, sample, episode_data, provenance):
        calls.append(sample["start"])
        target.mkdir(parents=True)
        for name in ("input.npz", "reference.npz", "config.json"):
            (target / name).write_text(json.dumps({"fixture": True, "gpu": runtime["gpus"]}))

    monkeypatch.setattr(module, f"write_{mode}_case", write_case)
    output = tmp_path / "prepared"
    for _ in range(2):
        result = module.prepare_full_test(tmp_path, assets, annotations, config, output, mode=mode)
        assert result["prepared_cases"] == 3 and not result["model_inference_performed"]
    assert calls == ([0, 0, 16] if mode == "short" else [0, 0, 0])
    plan = json.loads((output / "plan.json").read_text())
    assert len(plan["cases"]) == 3
    if mode == "short":
        assert plan["unavailable"]["no_official_short_window"] == ["000000"]
    else:
        assert not plan["unavailable"]
        assert plan["cases"][0]["id"] == "irasim_lt_000000_full_long_seed3407"
        meta = json.loads((output / "coverage.json").read_text())
        assert "all_indices_equal" not in meta and meta["short_window_indices_exact_equal"]
    (output / "plan.json").write_text("{}")
    with pytest.raises(ValueError, match="plan changed"):
        module.prepare_full_test(tmp_path, assets, annotations, config, output, mode=mode)
